use serde::{Serialize};
use serde::ser::{Serializer};

use ndarray::{Array1, Array2};

pub enum Variant {
    Int(i32),
    Float(f32),
    String(String)
}

pub type Num = f32;
pub type DataVec = Array1< Num >;
pub type WsMat = Array2< Num >;
pub type WsBlob = Vec< WsMat >;  
pub type Blob = Vec< DataVec >;

impl Serialize for Variant {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match &self {
            Variant::Int(val) => {
                serializer.serialize_i32(*val)
            },
            Variant::Float(val) => {
                serializer.serialize_f32(*val)
            },
            Variant::String(val) => {
                serializer.serialize_str(val.as_str())
            }
        }
    }
}
