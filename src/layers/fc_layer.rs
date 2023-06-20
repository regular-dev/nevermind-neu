use std::collections::HashMap;

use ndarray::{indices, Zip};

use log::debug;

use rand::{thread_rng, Rng, ThreadRng};

use std::ops::{Deref, DerefMut};

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use crate::cpu_params::*;
use crate::util::*;

use crate::util::{Variant, WithParams};

// Fully-connected layer
#[derive(Clone)]
pub struct FcLayer<T: Fn(f32) -> f32 + Clone, TD: Fn(f32) -> f32 + Clone> {
    pub lr_params: CpuParams,
    size: usize,
    dropout: f32,
    l2_regul: f32,
    l1_regul: f32,
    pub activation: Activation<T, TD>,
    rng: ThreadRng,
}

impl<T, TD> AbstractLayer for FcLayer<T, TD>
where
    T: Fn(f32) -> f32 + Sync + Clone + 'static,
    TD: Fn(f32) -> f32 + Sync + Clone + 'static,
{
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].get_2d_buf_t(TypeBuffer::Output);
        let inp_m = inp_m.borrow();
        let inp_m = inp_m.deref();

        let out_m = self.lr_params.get_2d_buf_t(TypeBuffer::Output);
        let mut out_m = out_m.borrow_mut();
        let out_m = out_m.deref_mut();

        let ws = self.lr_params.get_2d_buf_t(TypeBuffer::Weights);
        let ws = ws.borrow();
        let ws = ws.deref();

        let bias_out = self.lr_params.get_1d_buf_t(TypeBuffer::Bias);
        let bias_out = bias_out.borrow();
        let bias_out = bias_out.deref();

        let dropout_len = (self.size as f32 * self.dropout) as usize;
        let dropout_n = self.rng.gen_range(0, self.size - dropout_len as usize);
        let dropout_y = dropout_n + dropout_len;

        // for each input batch
        Zip::from(inp_m.rows())
            .and(out_m.rows_mut())
            .par_for_each(|inp_b, out_b| {
                // for each batch
                let mul_res = ws.dot(&inp_b);

                let mut counter_neu = 0;

                // for each neuron
                Zip::from(out_b)
                    .and(&mul_res)
                    .and(bias_out)
                    .for_each(|out_el, in_row, bias_el| {
                        // for each "neuron"
                        if counter_neu >= dropout_n && counter_neu < dropout_y {
                            // zero neuron
                            *out_el = 0.0;
                            counter_neu += 1;
                        } else {
                            *out_el = (self.activation.func)(in_row + bias_el);
                            counter_neu += 1;
                        }
                    });
            });

        debug!("[ok] HiddenLayer forward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn backward(&mut self, prev_input: ParamsBlob, next_input: ParamsBlob) -> LayerBackwardResult {
        let next_err_vals = next_input[0].get_2d_buf_t(TypeBuffer::NeuGrad);
        let next_err_vals = next_err_vals.borrow();
        let next_err_vals = next_err_vals.deref();

        let next_ws = next_input[0].get_2d_buf_t(TypeBuffer::Weights);
        let next_ws = next_ws.borrow();
        let next_ws = next_ws.deref();

        let self_err_vals = self.lr_params.get_2d_buf_t(TypeBuffer::NeuGrad);
        let mut self_err_vals = self_err_vals.borrow_mut();
        let self_err_vals = self_err_vals.deref_mut();

        let self_output = self.lr_params.get_2d_buf_t(TypeBuffer::Output);
        let self_output = self_output.borrow();
        let self_output = self_output.deref();

        let self_bias_grad = self.lr_params.get_1d_buf_t(TypeBuffer::BiasGrad);
        let mut self_bias_grad = self_bias_grad.borrow_mut();
        let self_bias_grad = self_bias_grad.deref_mut();

        let self_bias = self.lr_params.get_1d_buf_t(TypeBuffer::Bias);
        let mut self_bias = self_bias.borrow_mut();
        let self_bias = self_bias.deref_mut();

        Zip::from(self_err_vals.rows_mut())
            .and(next_err_vals.rows())
            .and(self_output.rows())
            .par_for_each(|err_val_r, next_err_val_r, output_r| {
                let mul_res = next_ws.t().dot(&next_err_val_r);

                Zip::from(err_val_r).and(output_r).and(&mul_res).for_each(
                    |err_val, output, col| {
                        *err_val = (self.activation.func_deriv)(*output) * col;
                    },
                );
            });

        debug!("[hidden layer] i am here 2");

        // calc per-weight gradient
        // for prev_layer :
        let prev_input = prev_input[0].get_2d_buf_t(TypeBuffer::Output);
        let prev_input = prev_input.borrow();
        let prev_input = prev_input.deref();

        let ws = self.lr_params.get_2d_buf_t(TypeBuffer::Weights);
        let ws = ws.borrow();
        let ws = ws.deref();

        let ws_grad = self.lr_params.get_2d_buf_t(TypeBuffer::WeightsGrad);
        let mut ws_grad = ws_grad.borrow_mut();
        let ws_grad = ws_grad.deref_mut();

        // calc grad for weights
        let ws_idxs = indices(ws_grad.dim());
        Zip::from(ws_grad)
            .and(ws)
            .and(ws_idxs)
            .par_for_each(|val_ws_grad, val_ws, ws_idx| {
                let self_neu_idx = ws_idx.0;
                let prev_neu_idx = ws_idx.1;

                let mut avg = 0.0;

                Zip::from(prev_input.column(prev_neu_idx))
                    .and(self_err_vals.column(self_neu_idx))
                    .for_each(|prev_val, err_val| {
                        avg += prev_val * err_val;
                    });

                avg = avg / prev_input.column(prev_neu_idx).len() as f32;

                let mut l2_penalty = 0.0;
                if self.l2_regul != 0.0 {
                    l2_penalty = self.l2_regul * val_ws;
                }

                let mut l1_penalty = 0.0;
                if self.l1_regul == 0.0 {
                    l1_penalty = self.l1_regul * sign(*val_ws);
                }

                *val_ws_grad = avg - l2_penalty - l1_penalty;
            });

        // calc grad for bias
        Zip::from(self_err_vals.columns())
            .and(self_bias_grad)
            .and(self_bias)
            .for_each(|err_vals, bias_grad, bias| {
                let grad = err_vals.mean().unwrap();

                let mut l2_penalty = 0.0;
                if self.l2_regul != 0.0 {
                    l2_penalty = self.l2_regul * *bias;
                }

                let mut l1_penalty = 0.0;
                if self.l1_regul == 0.0 {
                    l1_penalty = self.l1_regul * sign(*bias);
                }

                *bias_grad = grad - l2_penalty - l1_penalty;
            });

        debug!("[ok] HiddenLayer backward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn cpu_params(&self) -> Option<CpuParams> {
        Some(self.lr_params.clone())
    }

    fn set_cpu_params(&mut self, lp: CpuParams) {
        self.lr_params = lp;
    }

    fn layer_type(&self) -> &str {
        "FcLayer"
    }

    /// Carefull this method overwrites weights and all other params
    fn set_input_shape(&mut self, sh: &[usize]) {
        self.lr_params = CpuParams::new_with_bias(self.size, sh[0]);
    }

    fn size(&self) -> usize {
        self.size
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = Box::new(FcLayer::new(self.size, self.activation.clone()));
        copy_l.set_cpu_params(self.lr_params.copy());
        copy_l
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }
}

impl<T, TD> FcLayer<T, TD>
where
    T: Fn(f32) -> f32 + Clone,
    TD: Fn(f32) -> f32 + Clone,
{
    pub fn new(size: usize, activation: Activation<T, TD>) -> Self {
        Self {
            size,
            dropout: 0.0,
            lr_params: CpuParams::empty(),
            activation,
            l2_regul: 0.0,
            l1_regul: 0.0,
            rng: thread_rng(),
        }
    }

    pub fn new_box(size: usize, activation: Activation<T, TD>) -> Box<Self> {
        Box::new(FcLayer::new(size, activation))
    }

    pub fn dropout(mut self, val: f32) -> Self {
        self.dropout = val;
        self
    }

    pub fn set_dropout(&mut self, val: f32) {
        self.dropout = val;
    }

    pub fn l2_regularization(mut self, coef: f32) -> Self {
        self.l2_regul = coef;
        self
    }

    pub fn set_l2_regularization(&mut self, coef: f32) {
        self.l2_regul = coef;
    }

    pub fn l1_regularization(mut self, coef: f32) -> Self {
        self.l1_regul = coef;
        self
    }

    pub fn set_l1_regularization(&mut self, coef: f32) {
        self.l1_regul = coef;
    }
}

impl<T, TD> WithParams for FcLayer<T, TD>
where
    T: Fn(f32) -> f32 + Sync + Clone + 'static,
    TD: Fn(f32) -> f32 + Sync + Clone + 'static,
{
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg: HashMap<String, Variant> = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.size as i32));
        // cfg.insert("prev_size".to_owned(), Variant::Int(self.prev_size as i32));
        cfg.insert(
            "activation".to_owned(),
            Variant::String(self.activation.name.clone()),
        );
        cfg.insert("l2_regul".to_owned(), Variant::Float(self.l2_regul));
        cfg.insert("l1_regul".to_owned(), Variant::Float(self.l1_regul));
        cfg.insert("dropout".to_owned(), Variant::Float(self.dropout));

        cfg
    }

    fn set_cfg(&mut self, cfg: &HashMap<String, Variant>) {
        let mut size: usize = 0;

        if let Some(var_size) = cfg.get("size") {
            if let Variant::Int(var_size) = var_size {
                size = *var_size as usize;
            }
        }

        if size > 0 {
            self.size = size;
            self.lr_params = CpuParams::empty();
        }

        if let Some(dropout) = cfg.get("dropout") {
            if let Variant::Float(dropout) = dropout {
                self.dropout = *dropout;
            }
        }

        if let Some(l1_regul) = cfg.get("l1_regul") {
            if let Variant::Float(l1_regul) = l1_regul {
                self.l1_regul = *l1_regul;
            }
        }

        if let Some(l2_regul) = cfg.get("l2_regul") {
            if let Variant::Float(l2_regul) = l2_regul {
                self.l2_regul = *l2_regul;
            }
        }
    }
}
