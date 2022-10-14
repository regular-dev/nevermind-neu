use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use ndarray::Zip;

use log::debug;

use crate::learn_params::{LearnParams, ParamsBlob};


use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use crate::activation::Activation;

use crate::bias::{Bias, ConstBias};
use crate::util::Variant;


pub struct HiddenLayer<T: Fn(f32) -> f32, TD: Fn(f32) -> f32> {
    pub lr_params: LearnParams,
    pub size: usize,
    pub prev_size: usize,
    pub dropout: f32,
    pub bias: ConstBias,
    pub activation: Activation<T, TD>,
}

impl<T, TD> AbstractLayer for HiddenLayer<T, TD>
where
    T: Fn(f32) -> f32 + Sync,
    TD: Fn(f32) -> f32 + Sync,
{
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws = self.lr_params.ws.borrow();

        let mul_res = inp_m.deref() * &ws[0];

        let bias_out = self.bias.forward(&ws[1]);

        Zip::from(out_m.deref_mut())
            .and(mul_res.rows())
            .and(bias_out)
            .par_for_each(|out_el, in_row, bias_el| {
                *out_el = (self.activation.func)(in_row.sum() + bias_el);
            });

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
            .par_for_each(|err_val, output, col| {
                *err_val = (self.activation.func_deriv)(*output) * col.sum();
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
        cfg.insert("prev_size".to_owned(), Variant::Int(self.prev_size as i32));

        cfg
    }

    fn set_layer_cfg(&mut self, cfg: &HashMap<String, Variant>) {
        let (mut size, mut prev_size): (usize, usize) = (0, 0);

        if let Variant::Int(var_size) = cfg.get("size").unwrap() {
            size = *var_size as usize;
        }

        if let Variant::Int(var_prev_size) = cfg.get("prev_size").unwrap() {
            prev_size = *var_prev_size as usize;
        }

        if size > 0 && prev_size > 0 {
            self.size = size;
            self.prev_size = prev_size;
            self.lr_params = LearnParams::new_with_const_bias(self.size, self.prev_size);
            self.bias = ConstBias::new(self.size, 1.0);
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl<T, TD> HiddenLayer<T, TD>
where
    T: Fn(f32) -> f32,
    TD: Fn(f32) -> f32,
{
    pub fn new(size: usize, prev_size: usize, activation: Activation<T, TD>) -> Self {
        Self {
            size,
            prev_size,
            dropout: 1.0,
            lr_params: LearnParams::new_with_const_bias(size, prev_size),
            bias: ConstBias::new(size, 1.0),
            activation,
        }
    }

    
    pub fn dropout(mut self, val: f32) -> Self {
        self.dropout = val;
        self.lr_params.drop_ws(val);
        self
    }
}