use std::collections::HashMap;

use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use log::debug;

use super::learn_params::LearnParams;
use super::util::Num;

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use super::activation::{sigmoid, sigmoid_deriv};

use super::util::{Blob, DataVec, Variant, WsBlob, WsMat};

use rand::Rng;

pub struct HiddenLayer {
    pub lr_params: LearnParams,
    pub size: usize,
    pub prev_size: usize,
}

impl AbstractLayer for HiddenLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult {
        let inp_m = &input[0];
        let out_m = &mut self.lr_params.output[0];
        let ws_mat = &self.lr_params.ws[0];

        let mul_res = inp_m * ws_mat;

        for (idx, el) in out_m.indexed_iter_mut() {
            *el = sigmoid(mul_res.row(idx).sum());
        }

        debug!("[ok] HiddenLayer forward()");

        Ok(&self.lr_params.output)
    }

    fn backward(&mut self, input: &Blob, weights: &WsBlob) -> LayerBackwardResult {
        let inp_vec = &input[0];
        let err_vec = &mut self.lr_params.err_vals[0];
        let ws_vec = &weights[0];

        let err_mul = ws_vec * inp_vec;

        debug!("err mul row_count() - {}", err_mul.shape()[1]);

        for (idx, val) in err_vec.indexed_iter_mut() {
            *val = sigmoid_deriv(self.lr_params.output[0][idx]) * err_mul.column(idx).sum();
        }

        debug!("[ok] HiddenLayer backward()");

        Ok((&self.lr_params.err_vals, &self.lr_params.ws))
    }

    fn learn_params(&mut self) -> Option< &mut LearnParams > {
        Some(&mut self.lr_params)
    }

    // fn optimize(&mut self, prev_out: &Blob) -> &Blob {
    //     let prev_vec = &prev_out[0];

    //     for neu_idx in 0..self.size {
    //         for prev_idx in 0..self.prev_size {
    //             let cur_ws_idx = [neu_idx, prev_idx];
    //             // 0.2 - ALPHA
    //             self.ws[0][cur_ws_idx] += 0.2 * self.ws_delta[0][cur_ws_idx];
    //             // 0.2 - LEARNING RATE
    //             self.ws_delta[0][cur_ws_idx] = 0.2 * self.err_vals[0][neu_idx] * prev_vec[prev_idx];

    //             self.ws[0][cur_ws_idx] += self.ws_delta[0][cur_ws_idx];
    //         }
    //     }

    //     &self.output
    // }

    fn layer_type(&self) -> &str {
        "HiddenLayer"
    }

    fn layer_cfg(&self) -> HashMap<&str, Variant> {
        let mut cfg: HashMap<&str, Variant> = HashMap::new();

        cfg.insert(
            "layer_type",
            Variant::String(String::from(self.layer_type())),
        );
        cfg.insert("size", Variant::Int(self.size as i32));

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
            lr_params: LearnParams::new(size, prev_size),
        }
    }
}
