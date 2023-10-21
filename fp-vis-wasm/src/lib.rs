mod utils;
use std::fmt::Display;

use std::num::ParseIntError;
use std::ops::{Div, Mul};
use std::str::FromStr;

use std::cmp::Ordering;

use num_bigint::{BigInt, BigUint};
use num_bigint::{ToBigInt, ToBigUint};
use num_rational::Ratio;
use num_traits::float::FloatCore;
use num_traits::{One, PrimInt, Signed, Zero};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Sign {
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

#[derive(Debug, PartialEq, Eq)]
enum Exact {
    Finite(Sign, Ratio<BigUint>),
    Infinite(Sign),
    NaN,
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
            Exact::NaN => {
                write!(f, "NaN")?;
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
    fn to_float<F: FloatCore + Display>(&self) -> F {
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
            Exact::NaN => F::nan(),
        }
    }

    fn from_float<F: FloatCore>(v: F) -> Self {
        let sign = v.signum();
        if sign.is_nan() {
            return Self::NaN;
        }

        let sign = if sign == One::one() {
            Sign::Positive
        } else {
            Sign::Negative
        };

        if v.is_infinite() {
            return Self::Infinite(sign);
        }

        let v = Ratio::<BigInt>::from_float(v).unwrap().abs();
        let (n, d) = v.into();

        Self::Finite(
            sign,
            (n.to_biguint().unwrap(), d.to_biguint().unwrap()).into(),
        )
    }

    fn error(&self, other: &Self) -> Option<Ratio<BigInt>> {
        match (self, other) {
            (Exact::NaN, Exact::NaN) => Some(Zero::zero()),
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
            (Exact::NaN, _) => Exact::NaN,
            (_, Exact::NaN) => Exact::NaN,
            (Exact::Infinite(_), Exact::Infinite(_)) => Exact::NaN,

            (Exact::Infinite(s1), Exact::Finite(s2, _)) => Exact::Infinite(s1 * s2),
            (Exact::Finite(s1, _), Exact::Infinite(s2)) => Exact::Finite(s1 * s2, Zero::zero()),
            (Exact::Finite(s1, v1), Exact::Finite(s2, v2)) => {
                match (v1 == Zero::zero(), v2 == Zero::zero()) {
                    (true, true) => Exact::NaN,
                    (false, true) => Exact::Infinite(s1 * s2),
                    _ => Exact::Finite(s1 * s2, v1 / v2),
                }
            }
        }
    }
}

impl From<&Exact> for f64 {
    fn from(value: &Exact) -> Self {
        value.to_float()
    }
}

impl From<&Exact> for f32 {
    fn from(value: &Exact) -> Self {
        value.to_float()
    }
}

impl From<f64> for Exact {
    fn from(value: f64) -> Self {
        Self::from_float(value)
    }
}

impl From<f32> for Exact {
    fn from(value: f32) -> Self {
        Self::from_float(value)
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
enum ParseError {
    #[error("more than 2 e's encountered")]
    TooManyEs,
    #[error("invalid exponent: {0}")]
    InvalidExponent(ParseIntError),
    #[error("more than 2 points encountered")]
    TooManyPoints,
    #[error("empty number")]
    EmptyNumber,
    #[error("invalid digit: {0}")]
    InvalidDigit(u8),
}

impl FromStr for Exact {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some((n, d)) = s.split_once('/') {
            let n = Self::from_str(n)?;
            let d = Self::from_str(d)?;
            return Ok(n / d);
        }

        let s = s.trim();
        let bytes = s.as_bytes().to_ascii_lowercase();
        let mut rem = &bytes[..];

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
            b"nan" => return Ok(Self::NaN),
            _ => (),
        }

        let mut separated = rem.split(|b| b == &b'e').fuse();
        let before_e = separated.next().unwrap();
        let after_e = separated.next().unwrap_or(b"0");
        if separated.next().is_some() {
            return Err(ParseError::TooManyEs);
        }

        let exp: i32 = String::from_utf8(after_e.to_vec())
            .unwrap()
            .parse()
            .map_err(|e| ParseError::InvalidExponent(e))?;

        let mut separated = before_e.split(|b| b == &b'.').fuse();
        let before_dot = separated.next().unwrap();
        let after_dot = separated.next().unwrap_or(b"");
        if separated.next().is_some() {
            return Err(ParseError::TooManyPoints);
        }

        if before_dot == b"" && after_dot == b"" {
            return Err(ParseError::EmptyNumber);
        }

        fn parse_biguint(mut bytes: &[u8]) -> Result<BigUint, ParseError> {
            if bytes.is_empty() {
                bytes = b"0";
            }
            for b in bytes {
                if !(b'0'..=b'9').contains(b) {
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

#[derive(Serialize)]
struct FloatPart {
    bits: Vec<bool>,
    value: String,
}

#[derive(Serialize)]
struct FloatParts {
    sign: FloatPart,
    exponent: FloatPart,
    mantissa: FloatPart,
}

trait FloatCoreExt: FloatCore {
    const BITS: usize = std::mem::size_of::<Self>() * 8;

    const MANTISSA_BITS: usize;
    const EXPONENT_BITS: usize = Self::BITS - Self::MANTISSA_BITS - 1;

    type BitsT: PrimInt + From<u8>;
    fn to_bits(self) -> Self::BitsT;
    fn from_bits(v: Self::BitsT) -> Self;
}

impl FloatCoreExt for f64 {
    const MANTISSA_BITS: usize = f64::MANTISSA_DIGITS as usize - 1;

    type BitsT = u64;

    fn to_bits(self) -> Self::BitsT {
        self.to_bits()
    }

    fn from_bits(v: Self::BitsT) -> Self {
        Self::from_bits(v)
    }
}

impl FloatCoreExt for f32 {
    const MANTISSA_BITS: usize = f32::MANTISSA_DIGITS as usize - 1;

    type BitsT = u32;

    fn to_bits(self) -> Self::BitsT {
        self.to_bits()
    }

    fn from_bits(v: Self::BitsT) -> Self {
        Self::from_bits(v)
    }
}

impl FloatParts {
    fn from_float<F: FloatCoreExt>(v: F) -> Self {
        let bits = v.to_bits();
        let bit_arr: Vec<bool> = (0u8..F::BITS.try_into().unwrap())
            .rev()
            .map(|i| bits & (F::BitsT::from(1) << i.into()) != 0.into())
            .collect();

        let (mantissa, exponent, sign) = v.integer_decode();

        Self {
            sign: FloatPart {
                value: format_number(&sign.to_string()),
                bits: bit_arr[..1].to_vec(),
            },
            exponent: FloatPart {
                value: format_number(&exponent.to_string()),
                bits: bit_arr[1..F::EXPONENT_BITS + 1].to_vec(),
            },
            mantissa: FloatPart {
                value: format_number(&mantissa.to_string()),
                bits: bit_arr[F::EXPONENT_BITS + 1..].to_vec(),
            },
        }
    }
}

#[derive(Serialize)]
struct FInfo {
    value: String,
    category: String,
    error: String,
    parts: FloatParts,
}

impl FInfo {
    fn new<F: FloatCoreExt + Display>(exact: &Exact, v: F) -> Self {
        let v_exact = Exact::from_float(v);
        let error = exact
            .error(&v_exact)
            .map(|error| format_number(&format!("{:.100}", error)))
            .unwrap_or_else(|| "Infinity".to_string());

        Self {
            value: format_number(&format!("{:.100}", v)),
            category: format!("{:?}", v.classify()),
            error,
            parts: FloatParts::from_float(v),
        }
    }
}

#[derive(Serialize)]
struct FInfos {
    f64: FInfo,
    f32: FInfo,
}

#[derive(Serialize)]
struct Info {
    value: String,
    floats: FInfos,
}

#[wasm_bindgen]
pub struct FloatInfo {
    exact: Exact,
    f64: f64,
    f32: f32,
}

#[wasm_bindgen]
impl FloatInfo {
    #[wasm_bindgen(constructor)]
    pub fn new(v: &str) -> Option<FloatInfo> {
        let exact: Exact = v.parse().ok()?;

        let f64 = (&exact).into();
        let f32 = (&exact).into();

        Some(Self { exact, f64, f32 })
    }

    pub fn get_info(&self) -> Result<JsValue, JsValue> {
        let info = Info {
            value: self.exact.to_string(),
            floats: FInfos {
                f64: FInfo::new(&self.exact, self.f64),
                f32: FInfo::new(&self.exact, self.f32),
            },
        };
        Ok(serde_wasm_bindgen::to_value(&info)?)
    }

    fn set_f64(&mut self, f64: f64) {
        self.exact = f64.into();
        self.f64 = f64;
        self.f32 = (&self.exact).into();
    }

    fn set_f32(&mut self, f32: f32) {
        self.exact = f32.into();
        self.f64 = (&self.exact).into();
        self.f32 = f32;
    }

    pub fn toggle_bit_f64(&mut self, i: u8) {
        let mut bits = self.f64.to_bits();
        bits ^= 1 << (63 - i);
        self.set_f64(f64::from_bits(bits));
    }

    pub fn toggle_bit_f32(&mut self, i: u8) {
        let mut bits = self.f32.to_bits();
        bits ^= 1 << (31 - i);
        self.set_f32(f32::from_bits(bits));
    }

    pub fn add_to_bits_f64(&mut self, i: i32) {
        let bits = self.f64.to_bits();
        let bits = bits.wrapping_add_signed(i.into());
        self.set_f64(f64::from_bits(bits));
    }

    pub fn add_to_bits_f32(&mut self, i: i32) {
        let bits = self.f32.to_bits();
        let bits = bits.wrapping_add_signed(i);
        self.set_f32(f32::from_bits(bits));
    }
}

fn format_number(mut s: &str) -> String {
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

#[wasm_bindgen(start)]
fn start() {
    crate::utils::set_panic_hook();
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
        test_parse_ok("nan", NaN);
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
    fn test_format_number() {
        assert_eq!(format_number("123"), "123");
        assert_eq!(format_number("1234"), "1,234");
        assert_eq!(format_number("12345"), "12,345");
        assert_eq!(format_number("123456"), "123,456");
        assert_eq!(format_number("1234567"), "1,234,567");
    }
}
