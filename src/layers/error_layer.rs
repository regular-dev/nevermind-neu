use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use ndarray::Zip;
use ndarray::{array, Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use log::debug;

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use crate::activation::{sigmoid, sigmoid_deriv};
use crate::learn_params::{LearnParams, ParamsBlob};
use crate::util::{Blob, DataVec, Num, Variant, WsBlob, WsMat};

use rand::Rng;

#[derive(Default)]
pub struct ErrorLayer {
    pub error: f32,
    pub size: usize,
    pub prev_size: usize,
    pub lr_params: LearnParams,
}

impl AbstractLayer for ErrorLayer {
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws_mat = &self.lr_params.ws.borrow()[0];

        let mul_res = inp_m.deref() * ws_mat;

        for (idx, el) in out_m.indexed_iter_mut() {
            *el = sigmoid(mul_res.row(idx).sum());
        }

        debug!("[ok] ErrorLayer forward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn backward_output(
        &mut self,
        prev_input: ParamsBlob,
        expected_vec: &DataVec,
    ) -> LayerBackwardResult {
        let prev_input = &prev_input[0].output.borrow();
        // let mut self_err_vals = self.lr_params.err_vals.borrow_mut();
        // let self_output = self.lr_params.output.borrow();
        let mut self_err_vals = self.lr_params.err_vals.borrow_mut();
        let self_output = self.lr_params.output.borrow();

        Zip::from(self_err_vals.deref_mut())
            .and(self_output.deref())
            .and(expected_vec)
            .for_each(|err_val, output, expected| {
                *err_val = (expected - output) * sigmoid_deriv(*output);
            });

        // calc per-weight gradient, TODO : refactor code below
        // for prev_layer :
        let ws = self.lr_params.ws.borrow();
        let mut ws_grad = self.lr_params.ws_grad.borrow_mut();

        for neu_idx in 0..ws[0].shape()[0] {
            for prev_idx in 0..ws[0].shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                ws_grad[0][cur_ws_idx] = prev_input[prev_idx] * self_err_vals[neu_idx];
            }
        }

        debug!("[ok] ErrorLayer backward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn layer_type(&self) -> &str {
        "ErrorLayer"
    }

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.lr_params.clone())
    }

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let mut cfg: HashMap<String, Variant> = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.size as i32));
        cfg.insert("prev_size".to_owned(), Variant::Int(self.prev_size as i32));

        cfg
    }

    fn set_layer_cfg(&mut self, cfg: &HashMap<String, Variant>) {
        let (mut size, mut prev_size) : (usize, usize) = (0, 0);

        if let Variant::Int(var_size) = cfg.get("size").unwrap() {
            size = *var_size as usize;
        }

        if let Variant::Int(var_prev_size) = cfg.get("prev_size").unwrap() {
            prev_size = *var_prev_size as usize;
        }

        if size > 0 && prev_size > 0 {
            self.size = size;
            self.prev_size = prev_size;
            self.lr_params = LearnParams::new(self.size, self.prev_size);
        }
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
            lr_params: LearnParams::new(size, prev_size),
        }
    }

    fn count_euclidean_error(&mut self, expected_vals: &Vec<f32>) {
        // let self_lr_params = self.lr

        // let mut err: f32 = 0.0;
        // let out_vec = &self.lr_params.output;

        // for (idx, val) in out_vec.iter().enumerate() {
        //     err += (expected_vals[idx] - val).powf(2.0);
        // }

        // println!("Euclidean loss : {}", err);

        // self.error = err;
    }
}
