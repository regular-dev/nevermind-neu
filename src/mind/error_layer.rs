use std::collections::HashMap;

use ndarray::{Array2, Array1, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use log::{debug};

use super::abstract_layer::{
    AbstractLayer, LayerBackwardResult, LayerForwardResult,
};
use super::activation::{sigmoid, sigmoid_deriv};
use super::util::{Blob, Variant, DataVec, WsBlob, WsMat, Num};

use rand::Rng;

pub struct ErrorLayer {
    pub error: f32,
    pub err_vals: Blob,
    pub ws: WsBlob,
    pub ws_delta: WsBlob,
    pub size: usize,
    pub prev_size: usize,
    pub output: Blob,
}

impl AbstractLayer for ErrorLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult {
        let inp_m = &input[0];
        let out_m = &mut self.output[0];
        let ws_mat = &self.ws[0];

        let mul_res = inp_m * ws_mat;

        for (idx, el) in out_m.indexed_iter_mut() {
            *el = sigmoid( mul_res.row(idx).sum() );
        }

        debug!("[ok] ErrorLayer forward()");

        Ok(&self.output)
    }
    fn backward(&mut self, input: &Blob, _weights: &WsBlob) -> LayerBackwardResult {
        let expected_vec = &input[0];
        //self.count_euclidean_error(&expected_vec);

        let out_vec = &self.output[0];
        let err_vec = &mut self.err_vals[0];
        for (idx, _out_iter) in out_vec.iter().enumerate() {
            err_vec[idx] = (expected_vec[idx] - out_vec[idx]) * sigmoid_deriv(out_vec[idx]);
        }

        debug!("[ok] ErrorLayer backward()");

        Ok((&self.err_vals, &self.ws))
    }

    fn optimize(&mut self, prev_out: &Blob) -> &Blob {
        let prev_vec = &prev_out[0];

        for neu_idx in 0..self.size {
            for prev_idx in 0..self.prev_size {
                let cur_ws_idx = [neu_idx, prev_idx];

                // 0.2 - ALPHA
                self.ws[0][cur_ws_idx] += 0.2 * self.ws_delta[0][cur_ws_idx];
                // 0.2 - LEARNING RATE
                self.ws_delta[0][cur_ws_idx] = 0.2 * self.err_vals[0][neu_idx] * prev_vec[prev_idx];

                self.ws[0][cur_ws_idx] += self.ws_delta[0][cur_ws_idx];
            }
        }

        &self.output
    }

    fn layer_type(&self) -> &str {
        "ErrorLayer"
    }

    fn layer_cfg(&self) -> HashMap< &str, Variant > { 
        let mut cfg: HashMap<&str, Variant> = HashMap::new();

        cfg.insert("layer_type", Variant::String(String::from(self.layer_type())));
        cfg.insert("size", Variant::Int(self.size as i32));

        cfg
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl ErrorLayer {
    pub fn new(size: usize, prev_size: usize) -> Self {
        let out_vec = DataVec::zeros(size);
        let err_vals = DataVec::zeros(size);

        let ws = WsMat::random((size, prev_size), Uniform::new(-0.5, 0.5));
        let ws_delta = WsMat::zeros((size, prev_size));

        Self {
            size,
            prev_size,
            error: 0.0,
            output: vec![out_vec],
            ws: vec![ws],
            ws_delta: vec![ws_delta],
            err_vals: vec![err_vals],
        }
    }

    fn count_euclidean_error(&mut self, expected_vals: &Vec<f32>) {
        let mut err: f32 = 0.0;
        let out_vec = &self.output[0];

        for (idx, val) in out_vec.iter().enumerate() {
            err += (expected_vals[idx] - val) * (expected_vals[idx] - val); // TODO : pow(val, 2)
        }

        println!("Euclidean loss : {}", err);

        self.error = err;
    }
}
