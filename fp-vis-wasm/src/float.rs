use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{Debug, Display},
    marker::PhantomData,
    num::ParseIntError,
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

use bitvec::{
    access::BitSafeU8,
    prelude::{BitBox, Msb0, *},
    slice::BitSlice,
};
use bstr::ByteSlice;
use duplicate::duplicate_item;
use funty::{Floating, Integral};
use num_bigint::{BigInt, BigUint, ToBigInt, ToBigUint};
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{float::FloatCore, Num, One, Signed, Zero};

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

    fn minus_or_empty(self) -> &'static str {
        match self {
            Sign::Positive => "",
            Sign::Negative => "-",
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

#[derive(Debug, Clone, PartialEq, Eq)]
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

impl Exact {
    pub(crate) fn normalize_zero(self) -> Self {
        match self {
            Exact::Finite(_, value) if value.is_zero() => Exact::Finite(Sign::Positive, value),
            v => v,
        }
    }
}

fn to_u8(v: &BigUint) -> u8 {
    let digits = v.to_u32_digits();
    assert!(digits.len() <= 1);
    let digit = digits.first().cloned().unwrap_or(0);
    assert!(digit < 10);
    digit.try_into().unwrap()
}

fn calculate_repeating_decimal(v: Ratio<BigUint>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let ten = 10.to_biguint().unwrap();
    let (n, d) = v.into();

    let (int, mut n) = n.div_rem(&d);
    let int_digits = if int.is_zero() {
        vec![]
    } else {
        int.to_radix_be(10)
    };

    if n.is_zero() {
        return (int_digits, vec![], vec![]);
    }

    let mut frac_digits = vec![];
    let mut rem_to_index = HashMap::new();
    let mut i = 0;

    n *= &ten;
    let start_i = loop {
        let (digit, rem) = n.div_rem(&d);
        dbg!((&digit, &rem));
        frac_digits.push(to_u8(&digit));

        if rem.is_zero() {
            return (int_digits, frac_digits, vec![]);
        }

        if let Some(j) = rem_to_index.insert(n.clone(), i) {
            break j;
        }
        i += 1;
        n = rem * &ten;
    };

    frac_digits.pop();

    let repeat_digits = frac_digits.split_off(start_i);

    (int_digits, frac_digits, repeat_digits)
}

fn digit_to_char(digit: u8) -> char {
    (b'0' + digit) as char
}

fn digits_to_string(digits: &[u8]) -> String {
    digits.iter().cloned().map(digit_to_char).collect()
}

fn to_exact_decimal(value: &Ratio<BigUint>) -> String {
    let (int_digits, frac_digits, repeat_digits) = calculate_repeating_decimal(value.clone());

    let mut result = String::new();

    if int_digits.is_empty() {
        result.push('0');
    } else {
        result.push_str(&format_number(&digits_to_string(&int_digits)));
    }

    if frac_digits.is_empty() && repeat_digits.is_empty() {
        return result;
    }
    result.push('.');
    result.push_str(&digits_to_string(&frac_digits));

    result.push_str(r#"<span style="text-decoration: overline">"#);
    for d in repeat_digits {
        result.push(digit_to_char(d));
        //result.push('\u{0305}');
    }
    result.push_str("</span>");

    result
}

impl Exact {
    pub(crate) fn to_exact_decimal(&self) -> String {
        match self {
            Exact::Finite(sign, value) => {
                let sign_str = if *sign == Sign::Positive { "" } else { "-" };
                format!("{}{}", sign_str, to_exact_decimal(value))
            }
            Exact::Infinite(Sign::Positive) => "inf".to_string(),
            Exact::Infinite(Sign::Negative) => "-inf".to_string(),
            Exact::NaN(nan) => nan.to_string(),
        }
    }

    pub(crate) fn to_exact_hex_literal(&self) -> Option<String> {
        match self {
            Exact::Finite(sign, value) => {
                let d = value.denom();
                if d.count_ones() != 1 {
                    return None;
                };

                let num_pow2 = value.numer().trailing_zeros().unwrap_or(0);
                let num = value.numer() >> num_pow2;

                let log2 = i64::try_from(num_pow2).unwrap()
                    - i64::try_from(d.trailing_zeros().unwrap()).unwrap();

                Some(format!(
                    "{sign}0x{num:x}p{log2}",
                    sign = sign.minus_or_empty(),
                ))
            }
            Exact::Infinite(Sign::Positive) => Some("inf".to_string()),
            Exact::Infinite(Sign::Negative) => Some("-inf".to_string()),
            Exact::NaN(nan) => Some(nan.to_string()),
        }
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

    pub(crate) fn nearby_floats<F: FloatingExt + FloatCore + Display>(&self) -> Vec<(f64, Exact)> {
        match self {
            Exact::Finite(_, _) => {
                let v: F = self.to_float();
                if !Floating::is_finite(v) {
                    return vec![];
                }

                let mut floats = vec![];
                {
                    let mut v = v;
                    for _ in 0..2 {
                        if !Floating::is_finite(v) {
                            break;
                        }
                        v = v.prev();
                        floats.push(Exact::from_float(v));
                    }
                    floats.reverse();
                }

                floats.push(Exact::from_float(v));

                {
                    let mut v = v;
                    for _ in 0..2 {
                        if !Floating::is_finite(v) {
                            break;
                        }
                        v = v.next();
                        floats.push(Exact::from_float(v));
                    }
                }

                let d_neg = self.clone() - floats[0].clone();
                let d_pos = floats.last().unwrap().clone() - self.clone();

                let d_max = match d_neg.partial_cmp(&d_pos).unwrap() {
                    Ordering::Less | Ordering::Equal => d_pos,
                    Ordering::Greater => d_neg,
                };
                floats
                    .into_iter()
                    .map(|v| (((v.clone() - self.clone()) / d_max.clone()).to_float(), v))
                    .collect()
            }
            Exact::Infinite(_sign) => {
                vec![]
            }
            Exact::NaN(_) => vec![],
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
}

impl Neg for Exact {
    type Output = Exact;

    fn neg(self) -> Self::Output {
        match self {
            Exact::Finite(s, v) => Exact::Finite(s.flip(), v),
            Exact::Infinite(s) => Exact::Infinite(s.flip()),
            nan @ Exact::NaN(_) => nan,
        }
    }
}

impl Add for Exact {
    type Output = Exact;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Exact::NaN(nan1), Exact::NaN(nan2)) => Exact::NaN(nan1.min(nan2)),
            (nan @ Exact::NaN(_), _) => nan,
            (_, nan @ Exact::NaN(_)) => nan,
            (Exact::Infinite(s1), Exact::Infinite(s2)) => {
                if s1 == s2 {
                    Exact::Infinite(s1)
                } else {
                    Exact::NaN(NaN::default())
                }
            }
            (Exact::Infinite(s), Exact::Finite(_, _)) => Exact::Infinite(s),
            (Exact::Finite(_, _), Exact::Infinite(s)) => Exact::Infinite(s),
            (Exact::Finite(s1, v1), Exact::Finite(s2, v2)) => {
                if s1 == s2 {
                    return Exact::Finite(s1, v1 + v2);
                }

                if v1 >= v2 {
                    Exact::Finite(s1, v1 - v2)
                } else {
                    Exact::Finite(s2, v2 - v1)
                }
            }
        }
    }
}

impl Sub for Exact {
    type Output = Exact;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
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

impl PartialOrd for Exact {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Exact::NaN(_), _) => None,
            (_, Exact::NaN(_)) => None,
            (Exact::Infinite(s1), Exact::Infinite(s2)) => Some(s1.cmp(s2)),
            (Exact::Infinite(s1), Exact::Finite(_, _)) => {
                if s1 == &Sign::Positive {
                    Some(Ordering::Greater)
                } else {
                    Some(Ordering::Less)
                }
            }
            (Exact::Finite(_, _), Exact::Infinite(s1)) => {
                if s1 == &Sign::Positive {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            }
            (Exact::Finite(s1, v1), Exact::Finite(s2, v2)) => {
                if v1.is_zero() && v2.is_zero() {
                    return Some(Ordering::Equal);
                }

                match s1.cmp(s2) {
                    Ordering::Equal => {
                        let mut r = v1.cmp(v2);
                        if s1 == &Sign::Negative {
                            r = r.reverse();
                        }
                        Some(r)
                    }
                    Ordering::Less => Some(Ordering::Less),
                    Ordering::Greater => Some(Ordering::Greater),
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

fn parse_float_literal(bytes: &[u8], radix: u32) -> Result<Ratio<BigUint>, ParseError> {
    let (before_dot, after_dot) =
        split_1_or_2(bytes, b'.').map_err(|_| ParseError::TooManyPoints)?;
    let after_dot = after_dot.unwrap_or(b"");

    if before_dot == b"" && after_dot == b"" {
        return Err(ParseError::EmptyNumber);
    }

    fn parse_biguint(mut bytes: &[u8], radix: u32) -> Result<BigUint, ParseError> {
        if bytes.is_empty() {
            bytes = b"0";
        }
        for b in bytes {
            if !b.is_ascii_hexdigit() {
                return Err(ParseError::InvalidDigit(*b));
            }
        }

        Ok(BigUint::from_str_radix(bytes.to_str().unwrap(), radix).unwrap())
    }

    let integer_part = parse_biguint(before_dot, radix)?;
    let fractional_part = parse_biguint(after_dot, radix)?;

    let denominator = 10u8
        .to_biguint()
        .unwrap()
        .pow(after_dot.len().try_into().unwrap());
    let numerator = integer_part * &denominator + fractional_part;

    Ok(Ratio::new(numerator, denominator))
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum ParseError {
    #[error("more than 1 '/' character encountered")]
    TooManySlashes,
    #[error("more than 1 e encountered")]
    TooManyEs,
    #[error("invalid exponent: {0}")]
    InvalidExponent(ParseIntError),
    #[error("more than 1 '.' character encountered")]
    TooManyPoints,
    #[error("empty number")]
    EmptyNumber,
    #[error("invalid digit: {0}")]
    InvalidDigit(u8),

    #[error("missing 0x")]
    Missing0x,
    #[error("more than 1 '0x' encountered")]
    TooMany0xs,
    #[error("missing p")]
    MissingP,
    #[error("more than 1 p's encountered")]
    TooManyPs,
    #[error("invalid sign: {0}")]
    InvalidSign(u8),

    #[error("{0}")]
    ParseIntError(#[from] ParseIntError),
}

struct TooManySplits;

enum Split2Error {
    TooFewSplits,
    TooManySplits,
}

fn split_2<'a, 'b>(
    bytes: &'a [u8],
    split_by: &'b [u8],
) -> Result<(&'a [u8], &'a [u8]), Split2Error> {
    let mut split = bytes.split_str(split_by);
    let res = (
        split.next().ok_or(Split2Error::TooFewSplits)?,
        split.next().ok_or(Split2Error::TooFewSplits)?,
    );
    if split.next().is_some() {
        return Err(Split2Error::TooManySplits);
    }
    Ok(res)
}

fn i64_from_bytes(bytes: &[u8]) -> Result<i64, ParseError> {
    for b in bytes {
        if !b.is_ascii() {
            return Err(ParseError::InvalidDigit(*b));
        }
    }

    Ok(i64::from_str(bytes.to_str().unwrap())?)
}

impl Exact {
    fn try_from_hex_literal(bytes: &[u8]) -> Result<Self, ParseError> {
        let (before_x, after_x) = match split_2(bytes, b"0x") {
            Ok(v) => v,
            Err(Split2Error::TooFewSplits) => return Err(ParseError::Missing0x),
            Err(Split2Error::TooManySplits) => return Err(ParseError::TooMany0xs),
        };
        let (before_p, after_p) = match split_2(after_x, b"p") {
            Ok(v) => v,
            Err(Split2Error::TooFewSplits) => return Err(ParseError::MissingP),
            Err(Split2Error::TooManySplits) => return Err(ParseError::TooManyPs),
        };

        let sign = match before_x {
            b"" | b"+" => Sign::Positive,
            b"-" => Sign::Negative,
            _ => return Err(ParseError::InvalidSign(before_x[0])),
        };
        let mut value = parse_float_literal(before_p, 16)?;

        let log2 = i64_from_bytes(after_p)?;
        let mut abs_p = BigUint::zero();
        abs_p.set_bit(log2.abs_diff(0), true);
        let mut exp: Ratio<BigUint> = abs_p.into();
        if log2 < 0 {
            exp = exp.recip();
        }
        value *= exp;

        Ok(Exact::Finite(sign, value))
    }
}

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
        let bytes: Vec<u8> = bytes
            .into_iter()
            .filter(|&&b| !(b.is_ascii_whitespace() || b == b','))
            .map(|b| b.to_ascii_lowercase())
            .collect();
        let bytes = bytes.as_slice();

        let (before_slash, after_slash) =
            split_1_or_2(bytes, b'/').map_err(|_| ParseError::TooManySlashes)?;

        if let Some(after_slash) = after_slash {
            let n: Self = before_slash.try_into()?;
            let d: Self = after_slash.try_into()?;
            return Ok(n / d);
        }

        if let Ok(v) = Exact::try_from_hex_literal(bytes) {
            return Ok(v);
        };

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
            b"nan(quiet)" => {
                return Ok(Self::NaN(
                    NaN::new(Sign::Positive, NaNType::Quiet, 0).unwrap(),
                ))
            }
            b"nan(signaling)" => {
                return Ok(Self::NaN(
                    NaN::new(Sign::Positive, NaNType::Signaling, 1).unwrap(),
                ))
            }
            _ => (),
        }

        let (before_e, after_e) = split_1_or_2(rem, b'e').map_err(|_| ParseError::TooManyEs)?;
        let after_e = after_e.unwrap_or(b"0");

        let exp: i32 = String::from_utf8(after_e.to_vec())
            .unwrap()
            .parse()
            .map_err(ParseError::InvalidExponent)?;

        let mut rat = parse_float_literal(before_e, 10)?;
        rat *= Ratio::from(10u8.to_biguint().unwrap()).pow(exp);

        Ok(Self::Finite(sign, rat))
    }
}

impl FromStr for Exact {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
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

pub(crate) trait FloatingExt: Floating + FloatCore {
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

    fn min_positive_subnormal() -> Self {
        Self::from_bits(Self::Raw::ONE)
    }

    fn prev(self) -> Self {
        if self == Zero::zero() {
            return -Self::min_positive_subnormal();
        }
        let bits = self.to_bits();
        let bits = bits.wrapping_add(Self::Raw::ONE);
        Self::from_bits(bits)
    }

    fn next(self) -> Self {
        if self == Zero::zero() {
            return Self::min_positive_subnormal();
        }
        let bits = self.to_bits();
        let bits = bits.wrapping_add(Self::Raw::ONE);
        Self::from_bits(bits)
    }

    fn to_hex_literal(self) -> String {
        let (mut mantissa, mut exponent, sign) = self.integer_decode();
        let mantissa_pow2 = mantissa.trailing_zeros();
        mantissa >>= mantissa_pow2;
        exponent += i16::try_from(mantissa_pow2).unwrap();
        let sign_str = match sign {
            1 => "",
            -1 => "-",
            _ => unreachable!(),
        };
        format!("{sign_str}0x{mantissa:x}p{exponent}")
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
    use Exact::*;
    use Sign::*;

    use super::*;

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
        test_parse_ok("nan (quiet)", Exact::NaN(super::NaN::default()));
        test_parse_ok(
            "nan (signaling)",
            Exact::NaN(super::NaN::new(Sign::Positive, NaNType::Signaling, 1).unwrap()),
        );
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

    fn cmp_float(v1: f64, v2: f64) -> Option<Ordering> {
        Exact::from_float(v1).partial_cmp(&Exact::from_float(v2))
    }

    #[test]
    fn test_exact_ord_zero() {
        for v1 in [-0.0, 0.0] {
            for v2 in [-0.0, 0.0] {
                assert_eq!(cmp_float(v1, v2), Some(Ordering::Equal));
            }
        }
    }

    #[test]
    fn test_exact_ord() {
        let sorted = [
            -f64::INFINITY,
            -f64::MAX,
            -1.0,
            -f64::MIN_POSITIVE,
            -f64::min_positive_subnormal(),
            0.0,
            f64::min_positive_subnormal(),
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
            f64::INFINITY,
        ];

        for (i1, v1) in sorted.iter().enumerate() {
            for (i2, v2) in sorted.iter().enumerate() {
                assert_eq!(v1.partial_cmp(v2), Some(i1.cmp(&i2)));
            }
        }
    }

    #[test]
    fn test_prev_next() {
        assert_eq!(0.0f64.prev(), -f64::min_positive_subnormal());
        assert_eq!(0.0f64.next(), f64::min_positive_subnormal());
        assert_eq!((-0.0f64).prev(), -f64::min_positive_subnormal());
        assert_eq!((-0.0f64).next(), f64::min_positive_subnormal());
    }

    #[test]
    fn test_prev_monotone() {
        let mut v = 0.0;
        for _ in 0..100 {
            assert!(v.prev() < v);
            v = v.prev();
        }
    }

    #[test]
    fn test_next_monotone() {
        let mut v = 0.0;
        for _ in 0..100 {
            assert!(v.next() > v);
            v = v.next();
        }
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

    fn test_repeated_decimal((n, d): (u64, u64), expected: (Vec<u8>, Vec<u8>, Vec<u8>)) {
        assert_eq!(
            calculate_repeating_decimal(Ratio::new(
                n.to_biguint().unwrap(),
                d.to_biguint().unwrap()
            )),
            expected
        );
    }

    #[test]
    fn test_repeated_decimal_zero() {
        test_repeated_decimal((0, 1), (vec![], vec![], vec![]));
    }

    #[test]
    fn test_repeated_decimal_one() {
        test_repeated_decimal((1, 1), (vec![1], vec![], vec![]));
    }

    #[test]
    fn test_repeated_decimal_one_8th() {
        test_repeated_decimal((1, 8), (vec![], vec![1, 2, 5], vec![]));
    }

    #[test]
    fn test_repeated_decimal_one_third() {
        test_repeated_decimal((1, 3), (vec![], vec![], vec![3]));
    }

    #[test]
    fn test_repeated_decimal_one_30th() {
        test_repeated_decimal((1, 30), (vec![], vec![0], vec![3]));
    }

    #[test]
    fn test_repeated_decimal_100_over_3() {
        test_repeated_decimal((100, 3), (vec![3, 3], vec![], vec![3]));
    }

    #[test]
    fn test_repeated_decimal_one_19th() {
        assert_eq!(
            calculate_repeating_decimal(Ratio::new(
                1.to_biguint().unwrap(),
                19.to_biguint().unwrap()
            )),
            (
                vec![],
                vec![],
                vec![0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7, 3, 6, 8, 4, 2, 1]
            )
        );
    }

    #[test]
    fn test_repeated_decimal_7_over_12() {
        assert_eq!(
            calculate_repeating_decimal(Ratio::new(
                7.to_biguint().unwrap(),
                12.to_biguint().unwrap()
            )),
            (vec![], vec![5, 8], vec![3],)
        );
    }

    fn test_hex_roundtrip(input: &str, expected_output: &str) {
        let v = Exact::try_from_hex_literal(input.as_bytes()).unwrap();
        dbg!(&v);
        let output = v.to_exact_hex_literal().unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_hex_0x0p0() {
        test_hex_roundtrip("0x0p0", "0x0p0");
    }

    #[test]
    fn test_hex_0xap0() {
        test_hex_roundtrip("0xap0", "0x5p1");
    }

    #[test]
    fn test_hex_0xap_1() {
        test_hex_roundtrip("0xap-1", "0x5p0");
    }

    #[test]
    fn test_hex_0x1p10() {
        test_hex_roundtrip("0x1p10", "0x1p10");
    }

    #[test]
    fn test_hex_0x1p_1() {
        test_hex_roundtrip("0x1p-1", "0x1p-1");
    }

    #[test]
    fn test_nearby_zero() {
        let nearby = Exact::from_float(0.0).nearby_floats::<f64>();
        for (f, _) in nearby {
            assert!(f.abs() <= 1.0);
        }
    }

    #[test]
    fn test_nearby_neg_zero() {
        let nearby = Exact::from_float(-0.0).nearby_floats::<f64>();
        for (f, _) in nearby.iter() {
            assert!(f.abs() <= 1.0, "{nearby:#?}");
        }
    }
}
