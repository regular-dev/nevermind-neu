use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use ndarray::Zip;
use ndarray::{array, Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use log::debug;

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use crate::activation::{sigmoid, sigmoid_deriv, Activation};
use crate::learn_params::{LearnParams, ParamsBlob};
use crate::util::{Blob, DataVec, Num, Variant, WsBlob, WsMat};

use rand::Rng;

pub struct ErrorLayer<T: Fn(f32) -> f32, TD: Fn(f32) -> f32 > {
    pub error: f32,
    pub size: usize,
    pub prev_size: usize,
    pub lr_params: LearnParams,
    pub activation: Activation<T, TD>
}

impl<T, TD> AbstractLayer for ErrorLayer<T, TD>
where
    T: Fn(f32) -> f32 + Sync,
    TD: Fn(f32) -> f32 + Sync,
{
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws_mat = &self.lr_params.ws.borrow()[0];

        let mul_res = inp_m.deref() * ws_mat;

        Zip::from(out_m.deref_mut()).and(mul_res.rows()).par_for_each(
            |out_el, in_row| {
                *out_el = (self.activation.func)(in_row.sum());
            }
        );

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
            .par_for_each(|err_val, output, expected| {
                *err_val = (expected - output) * (self.activation.func_deriv)(*output);
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

impl<T, TD> ErrorLayer<T, TD>
where
    T: Fn(f32) -> f32,
    TD: Fn(f32) -> f32,
{
    pub fn new(size: usize, prev_size: usize, activation: Activation<T, TD>) -> Self {
        Self {
            size,
            prev_size,
            error: 0.0,
            lr_params: LearnParams::new(size, prev_size),
            activation,
        }
    }
}
