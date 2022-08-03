use std::collections::HashMap;

use ndarray::{ArrayViewMut1, Axis};

use super::abstract_layer::*;
use super::learn_params::LearnParams;
use super::util::{Blob, DataVec, Num, WsBlob};

pub struct ConstBias {
    pub lr: LearnParams,
    pub val: Num,
    pub output: Blob,
}

impl ConstBias {
    pub fn new(prev_size: usize, val: Num) -> Self {
        let mut lr = LearnParams::new(1, prev_size);

        lr.output[0] = DataVec::from_elem(1, val);

        let mut output = Blob::new();
        output.resize(1, DataVec::from_elem(0, 0.0));

        Self { lr, val, output: output }
    }

    // TODO : impl sized ConstBias -> [1.0, 1.0, 1.0] with weights.
}

impl AbstractLayer for ConstBias {
    fn forward(&mut self, _input: &Blob) -> LayerForwardResult {
        let out_vec = &mut self.output[0];
        let ws_mat = &self.lr.ws[0];

        let mul = (self.val * ws_mat)
            .map_axis_mut(Axis(0), &|arr_view: ArrayViewMut1<Num>| arr_view.sum());

        *out_vec = mul;

        // for (idx, val) in out_vec.indexed_iter_mut() {
        // *val = mul.column(idx).sum();
        // }

        Ok(&self.output)
    }
    /// returns out_values and array of weights
    fn backward(&mut self, input: &Blob, weights: &WsBlob) -> LayerBackwardResult {


        Ok((&self.lr.err_vals, &self.lr.ws))
    }

    fn size(&self) -> usize {
        1
    }

    fn layer_type(&self) -> &str {
        "ConstBias"
    }

    fn learn_params(&mut self) -> Option<&mut LearnParams> {
        Some(&mut self.lr)
    }
}
