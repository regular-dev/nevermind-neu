use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use ndarray::Zip;

use log::{debug, info};

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use crate::activation::*;
use crate::learn_params::{LearnParams, ParamsBlob};
use crate::util::{Batch, DataVec, Variant};

#[derive(Clone)]
pub struct ErrorLayer<T: Fn(f32) -> f32 + Clone, TD: Fn(f32) -> f32 + Clone> {
    pub size: usize,
    pub prev_size: usize,
    pub lr_params: LearnParams,
    pub l2_regul: f32,
    pub l1_regul: f32,
    pub activation: Activation<T, TD>,
}

impl<T, TD> AbstractLayer for ErrorLayer<T, TD>
where
    T: Fn(f32) -> f32 + Sync + Clone + 'static,
    TD: Fn(f32) -> f32 + Sync + Clone + 'static,
{
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws = self.lr_params.ws.borrow();
        let ws0 = &ws[0]; // for neurons
        let bias_out = ws[1].column(0);

        Zip::from(inp_m.rows())
            .and(out_m.rows_mut())
            .par_for_each(|inp_r, out_b| { // for each batch
                let mul_res = ws0.dot(&inp_r);

                // for each neuron
                Zip::from(out_b)
                    .and(&mul_res)
                    .and(bias_out)
                    .for_each(|out_el, in_row, bias_el| { // for each "neuron"
                        *out_el = (self.activation.func)(*in_row + bias_el);
                    });
            });

        debug!("[ok] ErrorLayer forward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn backward_output(
        &mut self,
        prev_input: ParamsBlob,
        expected_vec: Batch,
    ) -> LayerBackwardResult {
        let prev_input = &prev_input[0].output.borrow();
        let mut self_err_vals = self.lr_params.err_vals.borrow_mut();
        let self_output = self.lr_params.output.borrow();

        // for each batch
        Zip::from(self_err_vals.rows_mut())
            .and(expected_vec.rows())
            .and(self_output.rows())
            .par_for_each(|err_val_r, expected_r, output_r| {
                Zip::from(err_val_r).and(expected_r).and(output_r).for_each(
                    |err_val, expected, output| {
                        *err_val = (expected - output) * (self.activation.func_deriv)(*output);
                    },
                );
            });

        // calc per-weight gradient, TODO : refactor code below
        // for prev_layer :
        let ws = self.lr_params.ws.borrow();
        let mut ws_grad = self.lr_params.ws_grad.borrow_mut();

        for neu_idx in 0..ws[0].shape()[0] {
            for prev_idx in 0..ws[0].shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];

                let mut avg = 0.0;
                Zip::from(prev_input.column(prev_idx))
                    .and(self_err_vals.column(neu_idx))
                    .for_each(|prev_val, err_val| {
                        avg += prev_val * err_val;
                    });

                avg = avg / prev_input.column(prev_idx).len() as f32;

                ws_grad[0][cur_ws_idx] = avg;
            }
        }

        // grad for bias
        for neu_idx in 0..ws[1].shape()[0] {
            // self.size
            for prev_idx in 0..ws[1].shape()[1] {
                // 1
                let cur_ws_idx = [neu_idx, prev_idx];

                let mut avg = 0.0;
                Zip::from(self_err_vals.column(neu_idx)).for_each(|err_val| {
                    avg += err_val;
                });

                avg = avg / self_err_vals.column(prev_idx).len() as f32;

                ws_grad[1][cur_ws_idx] = avg;
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

    fn set_learn_params(&mut self, lp: LearnParams) {
        self.lr_params = lp;
    }

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let mut cfg: HashMap<String, Variant> = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.size as i32));
        cfg.insert("prev_size".to_owned(), Variant::Int(self.prev_size as i32));

        cfg.insert(
            "activation".to_owned(),
            Variant::String(self.activation.name.clone()),
        );

        cfg.insert("l2_regul".to_owned(), Variant::Float(self.l2_regul));
        cfg.insert("l1_regul".to_owned(), Variant::Float(self.l1_regul));

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
        }

        if let Variant::Float(l1_regul) = cfg.get("l1_regul").unwrap() {
            self.l1_regul = *l1_regul;
        }

        if let Variant::Float(l2_regul) = cfg.get("l2_regul").unwrap() {
            self.l2_regul = *l2_regul;
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = ErrorLayer::new(self.size, self.prev_size, self.activation.clone());
        copy_l.set_learn_params(self.lr_params.copy());
        Box::new(copy_l)
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }
}

impl<T, TD> ErrorLayer<T, TD>
where
    T: Fn(f32) -> f32 + Clone,
    TD: Fn(f32) -> f32 + Clone,
{
    pub fn new(size: usize, prev_size: usize, activation: Activation<T, TD>) -> Self {
        Self {
            size,
            prev_size,
            lr_params: LearnParams::new_with_const_bias(size, prev_size),
            activation,
            l1_regul: 0.0,
            l2_regul: 0.0,
        }
    }

    pub fn l2_regularization(mut self, coef: f32) -> Self {
        self.l2_regul = coef;
        self
    }

    pub fn l1_regularization(mut self, coef: f32) -> Self {
        self.l1_regul = coef;
        self
    }
}
