use std::fmt::Display;

use num_traits::float::FloatCore;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::float::{format_number, Exact, FloatBits, FloatingExt};

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

impl FloatParts {
    fn from_float<F: FloatingExt + FloatCore>(v: F) -> Self {
        let bits = FloatBits::from_float(v);
        let [sign_bit, exponent_bits, mantissa_bits] = bits.parts();

        let (mantissa, exponent, sign) = v.integer_decode();

        Self {
            sign: FloatPart {
                value: format_number(&sign.to_string()),
                bits: sign_bit.iter().by_vals().collect(),
            },
            exponent: FloatPart {
                value: format_number(&exponent.to_string()),
                bits: exponent_bits.iter().by_vals().collect(),
            },
            mantissa: FloatPart {
                value: format_number(&mantissa.to_string()),
                bits: mantissa_bits.iter().by_vals().collect(),
            },
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
            category: format!("{:?}", FloatCore::classify(v)),
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

#[derive(Serialize)]
struct Constant<F> {
    name: &'static str,
    value: F,
}

impl<F: FloatingExt + FloatCore> Constant<F> {
    fn new(name: &'static str, value: F) -> Self {
        Self { name, value }
    }
}

#[derive(Serialize)]
struct Constants {
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
            Constant::new("-∞", -F::infinity()),
            Constant::new("Min finite", -F::max_value()),
            Constant::new("-1", -F::one()),
            Constant::new("Max negative normal", -F::min_positive_value()),
            Constant::new("Max negative", -F::min_positive_subnormal()),
            Constant::new("-0", -F::zero()),
            Constant::new("+0", F::zero()),
            Constant::new("Min positive", F::min_positive_subnormal()),
            Constant::new("Min positive normal", F::min_positive_value()),
            Constant::new("1", F::one()),
            Constant::new("Max finite", F::max_value()),
            Constant::new("∞", F::infinity()),
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

    pub fn set_f64(&mut self, f64: f64) {
        self.exact = f64.into();
        self.f64 = f64;
        self.f32 = (&self.exact).into();
    }

    pub fn set_f32(&mut self, f32: f32) {
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

#[wasm_bindgen(start)]
fn start() {
    crate::utils::set_panic_hook();
}
