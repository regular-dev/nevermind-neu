use ndarray::Array;

use crate::util::{Blob, DataVec};

#[derive(Clone)]
pub struct DataBatch {
    pub input: DataVec,
    pub expected: DataVec,
}

impl DataBatch {
    pub fn new(input: Vec<f32>, expected: Vec<f32>) -> Self {
        Self {
            input: Array::from_vec(input),
            expected: Array::from_vec(expected),
        }
    }
}