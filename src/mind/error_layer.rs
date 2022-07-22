use std::collections::HashMap;

use super::abstract_layer::{
    AbstractLayer, LayerBackwardResult, LayerForwardResult,
};
use super::activation::{sigmoid, sigmoid_deriv};
use super::util::{Blob, Variant, DataVec};

use rand::Rng;

pub struct ErrorLayer {
    pub error: f32,
    pub error_vals: Blob,
    pub ws: Blob,
    pub ws_delta: Blob,
    pub size: usize,
    pub prev_size: usize,
    pub output: Blob,
}

impl AbstractLayer for ErrorLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult {
        let inp_vec = &input[0];
        let out_vec = &mut self.output[0];

        for idx_out in 0..out_vec.len() {
            let mut sum: f32 = 0.0;
            for (idx_in, val_in) in inp_vec.iter().enumerate() {
                sum += val_in * self.ws[0][idx_out * inp_vec.len() + idx_in];
            }
            let activated_val = sigmoid(sum);
            out_vec[idx_out] = activated_val;
        }

        Ok(&self.output)
    }
    fn backward(&mut self, input: &Blob, _weights: &Blob) -> LayerBackwardResult {
        let expected_vec = &input[0];
        //self.count_euclidean_error(&expected_vec);

        let out_vec = &self.output[0];
        let err_vec = &mut self.error_vals[0];
        for (idx, _out_iter) in out_vec.iter().enumerate() {
            err_vec[idx] = (expected_vec[idx] - out_vec[idx]) * sigmoid_deriv(out_vec[idx]);
        }

        Ok((&self.error_vals, &self.ws))
    }

    fn optimize(&mut self, prev_out: &Blob) -> &Blob {
        let prev_vec = &prev_out[0];

        for neu_idx in 0..self.size {
            for prev_idx in 0..self.prev_size {
                let ws_idx = neu_idx * self.prev_size + prev_idx;

                // 0.2 - ALPHA
                self.ws[0][ws_idx] += 0.2 * self.ws_delta[0][ws_idx];
                // 0.2 - LEARNING RATE
                self.ws_delta[0][ws_idx] = 0.2 * self.error_vals[0][neu_idx] * prev_vec[prev_idx];

                self.ws[0][ws_idx] += self.ws_delta[0][ws_idx];
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
        let mut out_vec: Vec<f32> = Vec::new();
        out_vec.resize(size, 0.0);

        let err_vals = out_vec.clone();

        let mut ws: Vec<Vec<f32>> = Vec::new();
        ws.resize(1, Vec::new());

        for i in &mut ws {
            i.resize_with(prev_size * size, || {
                rand::thread_rng().gen_range(-0.55, 0.55)
            });
        }

        let mut ws_delta = Vec::new();
        ws_delta.resize(prev_size * size, 0.0);

        Self {
            size,
            prev_size,
            error: 0.0,
            output: vec![out_vec],
            ws,
            ws_delta: vec![ws_delta],
            error_vals: vec![err_vals],
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
