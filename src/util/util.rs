use serde::{Serialize, Deserialize};
use serde::ser::{Serializer};

use std::rc::Rc;
use std::cell::RefCell;

use std::collections::HashMap;

use ndarray::{Array1, Array2};

pub type Float = f32;
pub type DataVec = Array1< Float >;
pub type DataVecPtr = Rc<RefCell<DataVec>>;
pub type Array2D = Array2< Float >;
pub type Array1D = Array1< Float >;
pub type WsMat = Array2< Float >;
pub type WsBlob = Vec< WsMat >;
pub type WsBlobPtr = Rc<RefCell<WsBlob>>;
pub type Blob<'a> = Vec< &'a DataVec >;
pub type Metrics = HashMap<String, f64>;

#[derive(Serialize, Deserialize)]
pub enum Variant {
    Int(i32),
    Float(f32),
    String(String),
}
