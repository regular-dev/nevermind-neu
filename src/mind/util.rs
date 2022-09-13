use serde::{Serialize, Deserialize};
use serde::ser::{Serializer};

use ndarray::{Array1, Array2};

#[derive(Serialize, Deserialize)]
pub enum Variant {
    Int(i32),
    Float(f32),
    String(String)
}

pub type Num = f32;
pub type DataVec = Array1< Num >;
pub type WsMat = Array2< Num >;
pub type WsBlob = Vec< WsMat >;  
pub type Blob<'a> = Vec< &'a DataVec >;