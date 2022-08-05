use std::collections::HashMap;

use ndarray::{ArrayViewMut1, Axis};

use super::abstract_layer::*;
use super::learn_params::LearnParams;
use super::util::{Blob, DataVec, Num, WsBlob, WsMat};


pub trait Bias {
    fn forward(&mut self, ws: &WsMat) -> &DataVec;
}

pub struct ConstBias {
    pub val: Num,
    pub output: DataVec,
}

impl ConstBias {
    pub fn new(size: usize, val: Num) -> Self {
        Self {
            val,
            output: DataVec::zeros(size),
        }
    }

    // TODO : impl sized ConstBias -> [1.0, 1.0, 1.0] with weights.
}

impl Bias for ConstBias {
    fn forward(&mut self, ws: &WsMat) -> &DataVec {

        let mul = (self.val * ws)
            .map_axis_mut(Axis(0), &|arr_view: ArrayViewMut1<Num>| arr_view.sum());

        self.output = mul;

        // for (idx, val) in out_vec.indexed_iter_mut() {
        // *val = mul.column(idx).sum();
        // }

        &self.output
    }
}
