use std::{fmt::Display, marker::PhantomData};

use bitvec::{order::Msb0, slice::BitSlice};
use funty::Floating;
use num_traits::float::FloatCore;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::float::{Exact, FloatBits, FloatingExt, format_number};

#[derive(Serialize)]
struct FloatPart {
    bits: Vec<bool>,
    raw_value: String,
    value: String,
}

fn bits_to_num(bits: &BitSlice<u8, Msb0>) -> String {
    let mut v = 0u64;
    for bit in bits {
        v *= 2;
        v += if *bit { 1 } else { 0 };
    }
    format_number(&v.to_string())
}

impl FloatPart {
    fn new(bits: &BitSlice<u8, Msb0>, value: impl Display) -> Self {
        Self {
            bits: bits.iter().by_vals().collect(),
            raw_value: bits_to_num(bits),
            value: format_number(&value.to_string()),
        }
    }
}

#[derive(Serialize)]
struct FloatParts {
    sign: FloatPart,
    exponent: FloatPart,
    mantissa: FloatPart,
}

impl FloatParts {
    fn from_float<F: FloatingExt + FloatCore>(v: F) -> Self {
        let bits = FloatBits::from_float(v);
        let [sign_bit, exponent_bits, mantissa_bits] = bits.parts();

        let (mantissa, exponent, sign) = v.integer_decode();

        Self {
            sign: FloatPart::new(sign_bit, sign),
            exponent: FloatPart::new(exponent_bits, exponent),
            mantissa: FloatPart::new(mantissa_bits, mantissa),
        }
    }
}

#[derive(Serialize)]
struct Value {
    fraction: String,
    decimal: String,
    hex_literal: Option<String>,
}

impl From<&Exact> for Value {
    fn from(value: &Exact) -> Self {
        Self {
            fraction: value.to_string(),
            decimal: value.to_exact_decimal(),
            hex_literal: value.to_exact_hex_literal(),
        }
    }
}

#[derive(Serialize)]
struct FInfo {
    hex: String,
    value: Value,
    category: String,
    error: Value,
    parts: FloatParts,
    nearby_floats: Vec<(f64, Value)>,
}

impl FInfo {
    fn new<F: FloatingExt + FloatCore + Display>(exact: &Exact, v: F) -> Self {
        let v_exact = Exact::from_float(v);
        let error = (v_exact.clone() - exact.clone()).normalize_zero();

        let nearby_floats = exact
            .nearby_floats::<F>()
            .into_iter()
            .map(|(f, v)| (f, (&v).into()))
            .collect();

        Self {
            hex: format!("0x{:0width$x}", v.to_bits(), width = F::BITS / 4),
            value: (&v_exact).into(),
            category: format!("{:?}", Floating::classify(v)),
            error: (&error).into(),
            parts: FloatParts::from_float(v),
            nearby_floats,
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
    value: Value,
    floats: FInfos,
}

#[derive(Serialize, Deserialize)]
pub struct U32Pair(u32, u32);

impl U32Pair {
    fn new(v: u64) -> Self {
        Self((v >> 32) as u32, v as u32)
    }
}

impl From<u64> for U32Pair {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl From<U32Pair> for u64 {
    fn from(value: U32Pair) -> Self {
        u64::from(value.0) << 32 | u64::from(value.1)
    }
}

#[derive(Serialize)]
struct Constant<F> {
    name: &'static str,
    bits: U32Pair,
    #[serde(skip)]
    _phantom: PhantomData<F>,
}

impl<F: FloatingExt + FloatCore> Constant<F> {
    fn new(name: &'static str, value: F) -> Self {
        let bits: u64 = value.to_bits().try_into().unwrap_or_else(|_| panic!());
        Self {
            name,
            bits: bits.into(),
            _phantom: PhantomData,
        }
    }
}

#[derive(Serialize)]
pub struct Constants {
    f64: Vec<Constant<f64>>,
    f32: Vec<Constant<f32>>,
}

impl Constants {
    pub fn new() -> Self {
        Self {
            f64: Self::all_constants(),
            f32: Self::all_constants(),
        }
    }

    fn all_constants<F: FloatingExt + FloatCore>() -> Vec<Constant<F>> {
        vec![
            Constant::new("-∞", -F::INFINITY),
            Constant::new("Min finite", -F::MAX),
            Constant::new("-1", -F::one()),
            Constant::new("Max negative normal", -F::MIN_POSITIVE),
            Constant::new("Max negative", -F::min_positive_subnormal()),
            Constant::new("-0", -F::zero()),
            Constant::new("+0", F::zero()),
            Constant::new("Min positive", F::min_positive_subnormal()),
            Constant::new("Min positive normal", F::MIN_POSITIVE),
            Constant::new("1", F::one()),
            Constant::new("Max finite", F::MAX),
            Constant::new("∞", F::INFINITY),
        ]
    }
}

#[wasm_bindgen]
pub struct FloatInfo {
    exact: Exact,
    f64: f64,
    f32: f32,
}

#[wasm_bindgen]
pub enum FloatType {
    F64 = "f64",
    F32 = "f32",
}

#[wasm_bindgen]
impl FloatInfo {
    #[wasm_bindgen(constructor)]
    pub fn new(v: &str) -> Result<FloatInfo, JsError> {
        let exact: Exact = v.parse()?;

        let f64 = (&exact).into();
        let f32 = (&exact).into();

        Ok(Self { exact, f64, f32 })
    }

    pub fn get_info(&self) -> Result<JsValue, JsValue> {
        let info = Info {
            value: (&self.exact).into(),
            floats: FInfos {
                f64: FInfo::new(&self.exact, self.f64),
                f32: FInfo::new(&self.exact, self.f32),
            },
        };
        Ok(serde_wasm_bindgen::to_value(&info)?)
    }

    pub fn constants(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&Constants::new())?)
    }

    fn set_bits_inner(&mut self, typ: FloatType, bits: u64) {
        match typ {
            FloatType::F64 => {
                let value = f64::from_bits(bits);
                self.exact = value.into();
                self.f64 = value;
                self.f32 = (&self.exact).into();
            }
            FloatType::F32 => {
                let value = f32::from_bits(bits as u32);
                self.exact = value.into();
                self.f64 = (&self.exact).into();
                self.f32 = value;
            }
            FloatType::__Invalid => unreachable!(),
        }
    }

    pub fn set_bits(&mut self, typ: FloatType, bits: JsValue) -> Result<(), JsValue> {
        let bits: U32Pair = serde_wasm_bindgen::from_value(bits)?;
        self.set_bits_inner(typ, bits.into());
        Ok(())
    }

    pub fn toggle_bit(&mut self, typ: FloatType, i: u8) {
        let bits = match typ {
            FloatType::F64 => self.f64.to_bits() ^ (1 << (63 - i)),
            FloatType::F32 => (self.f32.to_bits() ^ (1 << (31 - i))).into(),
            FloatType::__Invalid => unreachable!(),
        };
        self.set_bits_inner(typ, bits);
    }

    pub fn add_to_bits(&mut self, typ: FloatType, i: i8) {
        let bits = match typ {
            FloatType::F64 => self.f64.to_bits().wrapping_add_signed(i.into()),
            FloatType::F32 => self.f32.to_bits().wrapping_add_signed(i.into()).into(),
            FloatType::__Invalid => unreachable!(),
        };
        self.set_bits_inner(typ, bits);
    }
}

#[wasm_bindgen(start)]
fn start() {
    crate::utils::set_panic_hook();
}
