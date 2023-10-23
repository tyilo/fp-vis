use bitvec::access::BitSafeU8;
use bitvec::prelude::*;
use duplicate::duplicate_item;
use std::fmt::Debug;
use std::fmt::Display;
use std::marker::PhantomData;

use bitvec::slice::BitSlice;
use funty::Floating;
use std::num::ParseIntError;
use std::ops::{Div, Mul};
use std::str::FromStr;

use std::cmp::Ordering;

use bitvec::prelude::BitBox;
use bitvec::prelude::Msb0;
use num_bigint::{BigInt, BigUint};
use num_bigint::{ToBigInt, ToBigUint};
use num_rational::Ratio;
use num_traits::float::FloatCore;
use num_traits::{One, Signed, Zero};

pub(crate) fn format_number(mut s: &str) -> String {
    if let Some((a, b)) = s.split_once('/') {
        let a = format_number(a);
        let b = format_number(b);
        return format!("{a} / {b}");
    }

    if let Some((a, b)) = s.split_once('.') {
        let s = format_number(a);
        return format!("{s}.{b}");
    }

    let mut res = String::new();
    if let Some(t) = s.strip_prefix('-') {
        res.push('-');
        s = t;
    }

    let n = s.chars().count();
    for (i, c) in s.chars().enumerate() {
        if i != 0 && (n - 1 - i) % 3 == 2 {
            res.push(',');
        }
        res.push(c);
    }

    res
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum Sign {
    Positive,
    Negative,
}

impl Sign {
    fn flip(self) -> Self {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
        }
    }
}

impl Mul for Sign {
    type Output = Sign;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Sign::Positive, Sign::Positive) => Sign::Positive,
            (Sign::Positive, Sign::Negative) => Sign::Negative,
            (Sign::Negative, Sign::Positive) => Sign::Negative,
            (Sign::Negative, Sign::Negative) => Sign::Positive,
        }
    }
}

fn to_bigint_ratio(sign: Sign, v: Ratio<BigUint>) -> Ratio<BigInt> {
    let (n, d) = v.into();
    let mut n = n.to_bigint().unwrap();
    if sign == Sign::Negative {
        n = -n;
    }
    (n, d.to_bigint().unwrap()).into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum NaNType {
    Signaling,
    Quiet,
}

impl Mul for NaNType {
    type Output = NaNType;

    fn mul(self, rhs: Self) -> Self::Output {
        if self == NaNType::Quiet && rhs == NaNType::Quiet {
            NaNType::Quiet
        } else {
            NaNType::Signaling
        }
    }
}

impl Display for NaNType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            NaNType::Signaling => "signaling",
            NaNType::Quiet => "quiet",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NaN {
    pub(crate) typ: NaNType,
    pub(crate) sign: Sign,
    pub(crate) payload: u64,
}

impl NaN {
    fn new(sign: Sign, typ: NaNType, payload: u64) -> Option<Self> {
        if typ == NaNType::Signaling && payload == 0 {
            return None;
        }

        Some(Self { sign, typ, payload })
    }
}

impl Default for NaN {
    fn default() -> Self {
        Self {
            sign: Sign::Positive,
            typ: NaNType::Quiet,
            payload: 0,
        }
    }
}

impl Display for NaN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NaN ({})", self.typ)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Exact {
    Finite(Sign, Ratio<BigUint>),
    Infinite(Sign),
    NaN(NaN),
}

impl Display for Exact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exact::Finite(sign, value) => {
                if sign == &Sign::Negative {
                    write!(f, "-")?;
                }
                write!(f, "{}", format_number(&value.to_string()))?;
            }
            Exact::Infinite(Sign::Positive) => {
                write!(f, "inf")?;
            }
            Exact::Infinite(Sign::Negative) => {
                write!(f, "-inf")?;
            }
            Exact::NaN(nan) => {
                write!(f, "{}", nan)?;
            }
        }
        Ok(())
    }
}

fn closest_float<F: FloatCore + Display>(v: &Ratio<BigInt>) -> F {
    let (min, max) = if v.is_positive() {
        let max = F::max_value();
        let max_rat = Ratio::<BigInt>::from_float(max).unwrap();
        if v > &max_rat {
            return F::infinity();
        }
        (Zero::zero(), max)
    } else {
        let min = F::min_value();
        let min_rat = Ratio::<BigInt>::from_float(min).unwrap();
        if v < &min_rat {
            return F::neg_infinity();
        }
        (min, Zero::zero())
    };

    let mut min = min;
    let mut max = max;
    let one: F = One::one();
    let two: F = one + one;

    loop {
        let mid = min + (max - min) / two;
        eprintln!("min: {min}, mid: {mid}, max: {max}");
        if mid == min || mid == max {
            let min_rat = Ratio::<BigInt>::from_float(min).unwrap();
            let max_rat = Ratio::<BigInt>::from_float(max).unwrap();
            if (max_rat - v) < (v - min_rat) {
                return max;
            } else {
                return min;
            }
        }
        let mid_rat = Ratio::<BigInt>::from_float(mid).unwrap();
        match mid_rat.cmp(v) {
            Ordering::Equal => return mid,
            Ordering::Less => {
                min = mid;
            }
            Ordering::Greater => {
                max = mid;
            }
        }
    }
}

impl Exact {
    fn to_float<F: FloatingExt + FloatCore + Display>(&self) -> F {
        let zero: F = Zero::zero();
        match self {
            Exact::Finite(sign, value) => {
                if value == &Zero::zero() {
                    match sign {
                        Sign::Positive => zero,
                        Sign::Negative => -zero,
                    }
                } else {
                    let (n, d) = value.clone().into();
                    let mut signed_ratio: Ratio<BigInt> = (n.into(), d.into()).into();
                    if sign == &Sign::Negative {
                        signed_ratio = -signed_ratio;
                    }
                    closest_float(&signed_ratio)
                }
            }
            Exact::Infinite(Sign::Positive) => F::infinity(),
            Exact::Infinite(Sign::Negative) => F::neg_infinity(),
            Exact::NaN(nan) => F::specific_nan(nan.sign, nan.typ, nan.payload),
        }
    }

    pub(crate) fn from_float<F: FloatingExt + FloatCore>(v: F) -> Self {
        let sign = Floating::signum(v);
        if Floating::is_nan(sign) {
            return Self::NaN(v.to_nan().unwrap());
        }

        let sign = if sign == One::one() {
            Sign::Positive
        } else {
            Sign::Negative
        };

        if Floating::is_infinite(v) {
            return Self::Infinite(sign);
        }

        let v = Ratio::<BigInt>::from_float(v).unwrap().abs();
        let (n, d) = v.into();

        Self::Finite(
            sign,
            (n.to_biguint().unwrap(), d.to_biguint().unwrap()).into(),
        )
    }

    pub(crate) fn error(&self, other: &Self) -> Option<Ratio<BigInt>> {
        match (self, other) {
            (Exact::NaN(_), Exact::NaN(_)) => Some(Zero::zero()),
            (Exact::Infinite(s1), Exact::Infinite(s2)) => {
                if s1 == s2 {
                    Some(Zero::zero())
                } else {
                    None
                }
            }
            (Exact::Finite(s1, v1), Exact::Finite(s2, v2)) => {
                let r1 = to_bigint_ratio(*s1, v1.clone());
                let r2 = to_bigint_ratio(*s2, v2.clone());
                Some(r2 - r1)
            }
            _ => None,
        }
    }
}

impl Div for Exact {
    type Output = Exact;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Exact::NaN(nan1), Exact::NaN(nan2)) => Exact::NaN(nan1.min(nan2)),
            (nan @ Exact::NaN(_), _) => nan,
            (_, nan @ Exact::NaN(_)) => nan,
            (Exact::Infinite(_), Exact::Infinite(_)) => Exact::NaN(NaN::default()),

            (Exact::Infinite(s1), Exact::Finite(s2, _)) => Exact::Infinite(s1 * s2),
            (Exact::Finite(s1, _), Exact::Infinite(s2)) => Exact::Finite(s1 * s2, Zero::zero()),
            (Exact::Finite(s1, v1), Exact::Finite(s2, v2)) => {
                match (v1 == Zero::zero(), v2 == Zero::zero()) {
                    (true, true) => Exact::NaN(NaN::default()),
                    (false, true) => Exact::Infinite(s1 * s2),
                    _ => Exact::Finite(s1 * s2, v1 / v2),
                }
            }
        }
    }
}

#[duplicate_item(
  F;
  [f64];
  [f32];
)]
mod from_to_exact {
    use super::Exact;
    impl From<&Exact> for F {
        fn from(value: &Exact) -> Self {
            value.to_float()
        }
    }
    impl From<F> for Exact {
        fn from(value: F) -> Self {
            Self::from_float(value)
        }
    }
}

impl From<i64> for Exact {
    fn from(value: i64) -> Self {
        let v = value.to_bigint().unwrap();
        let (sign, mag) = v.into_parts();
        let sign = if sign == num_bigint::Sign::Minus {
            Sign::Negative
        } else {
            Sign::Positive
        };

        Self::Finite(sign, mag.into())
    }
}

impl From<(i64, i64)> for Exact {
    fn from((n, d): (i64, i64)) -> Self {
        let n = n.to_bigint().unwrap();
        let d = d.to_bigint().unwrap();

        Self::from(Ratio::<BigInt>::new(n, d))
    }
}

impl From<Ratio<BigInt>> for Exact {
    fn from(value: Ratio<BigInt>) -> Self {
        let (n, d) = value.into();
        let (n_sign, n_mag) = n.into_parts();
        let (d_sign, d_mag) = d.into_parts();

        let sign = match (n_sign, d_sign) {
            (num_bigint::Sign::Minus, num_bigint::Sign::Minus) => Sign::Positive,
            (num_bigint::Sign::Minus, _) => Sign::Negative,
            (_, num_bigint::Sign::Minus) => Sign::Negative,
            _ => Sign::Positive,
        };

        Self::Finite(sign, (n_mag, d_mag).into())
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum ParseError {
    #[error("more than 2 '/' characters encountered")]
    TooManySlashes,
    #[error("more than 2 e's encountered")]
    TooManyEs,
    #[error("invalid exponent: {0}")]
    InvalidExponent(ParseIntError),
    #[error("more than 2 '.' characters encountered")]
    TooManyPoints,
    #[error("empty number")]
    EmptyNumber,
    #[error("invalid digit: {0}")]
    InvalidDigit(u8),
}

struct TooManySplits;

fn split_1_or_2(bytes: &[u8], split_by: u8) -> Result<(&[u8], Option<&[u8]>), TooManySplits> {
    let mut split = bytes.split(|b| b == &split_by).fuse();
    let first = split.next().unwrap();
    let second = split.next();
    match split.next() {
        None => Ok((first, second)),
        Some(_) => Err(TooManySplits),
    }
}

impl TryFrom<&[u8]> for Exact {
    type Error = ParseError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let (before_slash, after_slash) =
            split_1_or_2(bytes, b'/').map_err(|_| ParseError::TooManySlashes)?;

        if let Some(after_slash) = after_slash {
            let n: Self = before_slash.try_into()?;
            let d: Self = after_slash.try_into()?;
            return Ok(n / d);
        }

        let mut rem = bytes;

        let mut sign = Sign::Positive;
        while let Some((b, r)) = rem.split_first() {
            match b {
                b'+' => (),
                b'-' => {
                    sign = sign.flip();
                }
                _ => break,
            }
            rem = r;
        }

        match rem {
            b"inf" | b"infinity" => return Ok(Self::Infinite(sign)),
            b"nan" => return Ok(Self::NaN(NaN::default())),
            _ => (),
        }

        let (before_e, after_e) = split_1_or_2(rem, b'e').map_err(|_| ParseError::TooManyEs)?;
        let after_e = after_e.unwrap_or(b"0");

        let exp: i32 = String::from_utf8(after_e.to_vec())
            .unwrap()
            .parse()
            .map_err(ParseError::InvalidExponent)?;

        let (before_dot, after_dot) =
            split_1_or_2(before_e, b'.').map_err(|_| ParseError::TooManyPoints)?;
        let after_dot = after_dot.unwrap_or(b"");

        if before_dot == b"" && after_dot == b"" {
            return Err(ParseError::EmptyNumber);
        }

        fn parse_biguint(mut bytes: &[u8]) -> Result<BigUint, ParseError> {
            if bytes.is_empty() {
                bytes = b"0";
            }
            for b in bytes {
                if !b.is_ascii_digit() {
                    return Err(ParseError::InvalidDigit(*b));
                }
            }

            Ok(BigUint::from_str(&String::from_utf8(bytes.to_vec()).unwrap()).unwrap())
        }

        let integer_part = parse_biguint(before_dot)?;
        let fractional_part = parse_biguint(after_dot)?;

        let denominator = 10u8
            .to_biguint()
            .unwrap()
            .pow(after_dot.len().try_into().unwrap());
        let numerator = integer_part * &denominator + fractional_part;

        let mut rat = Ratio::new(numerator, denominator);
        rat *= Ratio::from(10u8.to_biguint().unwrap()).pow(exp);

        Ok(Self::Finite(sign, rat))
    }
}

impl FromStr for Exact {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s
            .replace(|c: char| c.is_whitespace() || c == ',', "")
            .to_ascii_lowercase();
        s.as_bytes().try_into()
    }
}

#[derive(Debug)]
pub(crate) struct FloatBits<F: FloatingExt> {
    bits: BitBox<u8, Msb0>,
    _phantom: PhantomData<F>,
}

impl<F: FloatingExt> FloatBits<F> {
    fn zero() -> Self {
        Self {
            bits: bitvec![u8, Msb0; 0; F::BITS].into(),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn from_float(v: F) -> Self {
        Self {
            bits: BitBox::from_boxed_slice(v.to_boxed_be_bytes()),
            _phantom: PhantomData,
        }
    }

    fn to_float(&self) -> F {
        F::from_bits(self.bits.load_be())
    }

    #[duplicate_item(
        method       split_at       u8          reference(type);
        [parts]      [split_at]     [u8]        [& type];
        [parts_mut]  [split_at_mut] [BitSafeU8] [&mut type];
    )]
    #[allow(clippy::needless_arbitrary_self_type)]
    pub(crate) fn method(self: reference([Self])) -> [reference([BitSlice<u8, Msb0>]); 3] {
        let (sign_bit, rem) = self.bits.split_at(1);
        let (exponent_bits, mantissa_bits) = rem.split_at(F::EXPONENT_BITS);
        [sign_bit, exponent_bits, mantissa_bits]
    }
}

pub(crate) trait FloatingExt: Floating {
    fn to_boxed_be_bytes(self) -> Box<[u8]>;

    const BITS: usize = std::mem::size_of::<Self>() * 8;

    const MANTISSA_BITS: usize = Self::MANTISSA_DIGITS as usize - 1;
    const EXPONENT_BITS: usize = Self::BITS - Self::MANTISSA_BITS - 1;

    fn specific_nan(sign: Sign, typ: NaNType, payload: u64) -> Self {
        let mut bits = FloatBits::<Self>::zero();
        let [sign_bit, exponent_bits, mantissa_bits] = bits.parts_mut();
        let (nan_type_bit, payload_bits) = mantissa_bits.split_at_mut(1);

        if sign == Sign::Negative {
            sign_bit.set(0, true);
        }

        exponent_bits.fill(true);

        if typ == NaNType::Quiet {
            nan_type_bit.set(0, true);
        }

        payload_bits.store_be(payload);

        bits.to_float()
    }

    fn to_nan(self) -> Option<NaN> {
        let bits = FloatBits::from_float(self);
        let [sign_bit, exponent_bits, mantissa_bits] = bits.parts();
        let (nan_type_bit, payload_bits) = mantissa_bits.split_at(1);

        let sign = if sign_bit[0] {
            Sign::Negative
        } else {
            Sign::Positive
        };

        if !exponent_bits.all() {
            return None;
        }

        let typ = if nan_type_bit[0] {
            NaNType::Quiet
        } else {
            NaNType::Signaling
        };

        let payload: u64 = payload_bits.load_be();

        Some(NaN::new(sign, typ, payload).unwrap())
    }
}

impl FloatingExt for f64 {
    fn to_boxed_be_bytes(self) -> Box<[u8]> {
        Box::new(self.to_be_bytes())
    }
}

impl FloatingExt for f32 {
    fn to_boxed_be_bytes(self) -> Box<[u8]> {
        Box::new(self.to_be_bytes())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use Exact::*;
    use Sign::*;

    fn test_parse_ok<T: Into<Exact>>(s: &str, v: T) {
        assert_eq!(s.parse(), Ok(v.into()));
    }

    #[test]
    fn test_inf() {
        test_parse_ok("Infinity", Infinite(Positive));
        test_parse_ok("-Infinity", Infinite(Negative));
    }

    #[test]
    fn test_nan() {
        test_parse_ok("nan", Exact::NaN(super::NaN::default()));
    }

    #[test]
    fn test_pos_zero() {
        let pos_zero = Ok(Exact::Finite(Sign::Positive, Zero::zero()));

        assert_eq!("0".parse(), pos_zero);
        assert_eq!("0.0".parse(), pos_zero);
        assert_eq!(".0".parse(), pos_zero);
        assert_eq!("0.".parse(), pos_zero);
        assert_eq!(".000".parse(), pos_zero);
        assert_eq!("000.".parse(), pos_zero);
        assert_eq!("000.000".parse(), pos_zero);
    }

    #[test]
    fn test_neg_zero() {
        let neg_zero = Ok(Exact::Finite(Sign::Negative, Zero::zero()));

        assert_eq!("-0".parse(), neg_zero);
        assert_eq!("-0.0".parse(), neg_zero);
        assert_eq!("-.0".parse(), neg_zero);
        assert_eq!("-0.".parse(), neg_zero);
        assert_eq!("-.000".parse(), neg_zero);
        assert_eq!("-000.".parse(), neg_zero);
        assert_eq!("-000.000".parse(), neg_zero);
    }

    #[test]
    fn test_integer() {
        test_parse_ok("1", 1);
        test_parse_ok("2", 2);
        test_parse_ok("123", 123);
        test_parse_ok("999999", 999999);

        test_parse_ok("-1", -1);
        test_parse_ok("-2", -2);
        test_parse_ok("-123", -123);
        test_parse_ok("-999999", -999999);
    }

    #[test]
    fn test_fractional() {
        test_parse_ok("0.1", (1, 10));
        test_parse_ok("0.01", (1, 100));
        test_parse_ok("0.000001", (1, 1000000));

        test_parse_ok("0.123", (123, 1000));
        test_parse_ok("0.25", (1, 4));

        test_parse_ok("-0.1", (-1, 10));
        test_parse_ok("-0.01", (-1, 100));
        test_parse_ok("-0.000001", (-1, 1000000));

        test_parse_ok("-0.123", (-123, 1000));
        test_parse_ok("-0.25", (-1, 4));
    }

    #[test]
    fn test_mixed() {
        test_parse_ok("1.2", (12, 10));
        test_parse_ok("10.02", (1002, 100));
        test_parse_ok("1.002", (1002, 1000));
        test_parse_ok("100.2", (1002, 10));
    }

    #[test]
    fn test_to_f64_1() {
        let v: Exact = "1.0".parse().unwrap();
        let x: f64 = (&v).into();
        assert_eq!(x, 1.0);
    }

    #[test]
    fn test_to_f64_0_3() {
        let v: Exact = "0.3".parse().unwrap();
        let x: f64 = (&v).into();
        assert_eq!(x, 0.3);
    }

    #[test]
    fn test_to_f64_0_3_next() {
        let v: Exact = "0.30000000000000004".parse().unwrap();
        let x: f64 = (&v).into();
        assert_eq!(x, 0.30000000000000004);
    }

    #[test]
    fn test_to_f64_0() {
        let v: Exact = "0.0".parse().unwrap();
        let x: f64 = (&v).into();
        assert_eq!(x, 0.0);
    }

    #[test]
    fn test_to_f64_min_normal_positive() {
        let v: Exact = "2.2250738585072014e-308".parse().unwrap();
        let x: f64 = (&v).into();
        assert_eq!(x, 2.2250738585072014e-308);
    }

    #[test]
    fn test_to_f64_min_subnormal_positive() {
        let v: Exact = "5e-324".parse().unwrap();
        let x: f64 = (&v).into();
        assert_eq!(x, 5e-324);
    }

    #[test]
    fn nan_conversion() {
        let nan = super::NaN::new(Positive, NaNType::Quiet, 123).unwrap();

        let v = Exact::NaN(nan.clone());
        let f64 = v.to_float::<f64>();
        assert_eq!(f64.to_nan().as_ref(), Some(&nan));

        let f32 = v.to_float::<f32>();
        assert_eq!(f32.to_nan().as_ref(), Some(&nan));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number("123"), "123");
        assert_eq!(format_number("1234"), "1,234");
        assert_eq!(format_number("12345"), "12,345");
        assert_eq!(format_number("123456"), "123,456");
        assert_eq!(format_number("1234567"), "1,234,567");
    }
}
