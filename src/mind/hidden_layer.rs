use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use ndarray::Zip;
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use log::debug;

use super::learn_params::{LearnParams, ParamsBlob};
use super::util::Num;

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use super::activation::{sigmoid, sigmoid_deriv};
use super::bias::{Bias, ConstBias};
use super::util::{Blob, DataVec, Variant, WsBlob, WsMat};

use rand::Rng;

pub struct HiddenLayer {
    pub lr_params: LearnParams,
    pub size: usize,
    pub prev_size: usize,
    pub bias: ConstBias,
}

impl AbstractLayer for HiddenLayer {
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws = self.lr_params.ws.borrow();

        let mul_res = inp_m.deref() * &ws[0];

        let bias_out = self.bias.forward(&ws[1]);

        for (idx, el) in out_m.indexed_iter_mut() {
            *el = sigmoid(mul_res.row(idx).sum() + bias_out[idx]);
        }

        debug!("[ok] HiddenLayer forward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn backward(&mut self, prev_input: ParamsBlob, next_input: ParamsBlob) -> LayerBackwardResult {
        let next_err_vals = next_input[0].err_vals.borrow();
        let next_ws = next_input[0].ws.borrow();
        let mut self_err_vals = self.lr_params.err_vals.borrow_mut();
        let self_output = self.lr_params.output.borrow();

        let err_mul = &next_ws[0] * next_err_vals[0];

        Zip::from(self_err_vals.deref_mut())
            .and(self_output.deref())
            .and(err_mul.columns())
            .for_each(|err_val, output, col| {
                *err_val = sigmoid_deriv(*output) * col.sum();
            });

        // calc per-weight gradient, TODO : refactor code below
        // for prev_layer :
        let prev_input = prev_input[0].output.borrow();
        let ws = self.lr_params.ws.borrow();
        let mut ws_grad = self.lr_params.ws_grad.borrow_mut();

        for neu_idx in 0..ws[0].shape()[0] {
            for prev_idx in 0..ws[0].shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                ws_grad[0][cur_ws_idx] = prev_input[prev_idx] * self_err_vals[neu_idx];
            }
        }

        // for bias :
        for neu_idx in 0..ws[1].shape()[0] {
            for prev_idx in 0..ws[1].shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                ws_grad[1][cur_ws_idx] = self.bias.val * self_err_vals[neu_idx];
            }
        }

        debug!("[ok] HiddenLayer backward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.lr_params.clone())
    }

    fn layer_type(&self) -> &str {
        "HiddenLayer"
    }

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let mut cfg: HashMap<String, Variant> = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.size as i32));

        cfg
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl HiddenLayer {
    pub fn new(size: usize, prev_size: usize) -> Self {
        Self {
            size,
            prev_size,
            lr_params: LearnParams::new_with_const_bias(size, prev_size),
            bias: ConstBias::new(size, 1.0),
        }
    }
}
