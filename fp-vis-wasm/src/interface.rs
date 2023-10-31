use crate::float::format_number;
use crate::float::{Exact, FloatBits, FloatingExt};
use funty::Floating;
use std::fmt::Display;

use num_traits::float::FloatCore;
use serde::Serialize;
use wasm_bindgen::prelude::*;

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
}

impl From<&Exact> for Value {
    fn from(value: &Exact) -> Self {
        Self {
            fraction: value.to_string(),
            decimal: value.to_exact_decimal(),
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
}

impl FInfo {
    fn new<F: FloatingExt + FloatCore + Display>(exact: &Exact, v: F) -> Self {
        let v_exact = Exact::from_float(v);
        let error = v_exact.clone() - exact.clone();

        Self {
            hex: format!("0x{:0width$x}", v.to_bits(), width = F::BITS / 4),
            value: (&v_exact).into(),
            category: format!("{:?}", Floating::classify(v)),
            error: (&error).into(),
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
    value: Value,
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

#[wasm_bindgen(start)]
fn start() {
    crate::utils::set_panic_hook();
}
