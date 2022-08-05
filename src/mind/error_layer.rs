use std::collections::HashMap;

use ndarray::{Array2, Array1, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use log::{debug};

use super::abstract_layer::{
    AbstractLayer, LayerBackwardResult, LayerForwardResult,
};
use super::activation::{sigmoid, sigmoid_deriv};
use super::learn_params::LearnParams;
use super::util::{Blob, Variant, DataVec, WsBlob, WsMat, Num};

use rand::Rng;

pub struct ErrorLayer {
    pub error: f32,
    pub size: usize,
    pub prev_size: usize,
    pub lr_params: LearnParams,
}

impl AbstractLayer for ErrorLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult {
        let inp_m = input[0];
        let out_m = &mut self.lr_params.output;
        let ws_mat = &self.lr_params.ws[0];

        let mul_res = inp_m * ws_mat;

        for (idx, el) in out_m.indexed_iter_mut() {
            *el = sigmoid( mul_res.row(idx).sum() );
        }

        debug!("[ok] ErrorLayer forward()");

        Ok(&self.lr_params.output)
    }
    fn backward(&mut self, input: &Blob, _weights: &WsBlob) -> LayerBackwardResult {
        let expected_vec = input[0];
        //self.count_euclidean_error(&expected_vec);

        let out_vec = &self.lr_params.output;
        let err_vec = &mut self.lr_params.err_vals;
        for (idx, _out_iter) in out_vec.iter().enumerate() {
            err_vec[idx] = (expected_vec[idx] - out_vec[idx]) * sigmoid_deriv(out_vec[idx]);
        }

        debug!("[ok] ErrorLayer backward()");

        Ok((&self.lr_params.err_vals, &self.lr_params.ws))
    }

    fn layer_type(&self) -> &str {
        "ErrorLayer"
    }

    fn learn_params(&mut self) -> Option< &mut LearnParams > {
        Some(&mut self.lr_params)
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
        Self {
            size,
            prev_size,
            error: 0.0,
            lr_params: LearnParams::new(size, prev_size)
        }
    }

    fn count_euclidean_error(&mut self, expected_vals: &Vec<f32>) {
        let mut err: f32 = 0.0;
        let out_vec = &self.lr_params.output;

        for (idx, val) in out_vec.iter().enumerate() {
            err += (expected_vals[idx] - val).powf(2.0);
        }

        println!("Euclidean loss : {}", err);

        self.error = err;
    }
}
