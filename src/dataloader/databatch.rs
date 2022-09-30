use ndarray::Array;

use std::cell::RefCell;

use crate::util::{Blob, DataVecPtr, DataVec};


#[derive(Clone, Default)]
pub struct DataBatch {
    pub input: DataVecPtr,
    pub expected: DataVec,
}

impl DataBatch {
    pub fn new(input: Vec<f32>, expected: Vec<f32>) -> Self {
        Self {
            input: DataVecPtr::new(RefCell::new(Array::from_vec(input))),
            expected: Array::from_vec(expected),
        }
    }
}