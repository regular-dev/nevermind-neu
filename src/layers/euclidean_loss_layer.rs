use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use ndarray::Zip;

use log::{debug, info};

use crate::layers::*;
use crate::cpu_params::*;
use crate::util::*;

#[derive(Clone)]
pub struct EuclideanLossLayer<T: Fn(f32) -> f32 + Clone, TD: Fn(f32) -> f32 + Clone> {
    pub size: usize,
    pub lr_params: CpuParams,
    pub l2_regul: f32,
    pub l1_regul: f32,
    pub activation: Activation<T, TD>,
}

impl<T, TD> AbstractLayer for EuclideanLossLayer<T, TD>
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

        Zip::from(inp_m.rows())
            .and(out_m.rows_mut())
            .par_for_each(|inp_r, out_b| {
                // for each batch
                let mul_res = ws.dot(&inp_r);

                // for each neuron
                Zip::from(out_b)
                    .and(&mul_res)
                    .and(bias_out)
                    .for_each(|out_el, in_row, bias_el| {
                        // for each "neuron"
                        *out_el = (self.activation.func)(*in_row + bias_el);
                    });
            });

        debug!("[ok] ErrorLayer forward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn backward_output(
        &mut self,
        prev_input: ParamsBlob,
        expected_vec: Array2D,
    ) -> LayerBackwardResult {
        let prev_input = prev_input[0].get_2d_buf_t(TypeBuffer::Output);
        let prev_input = prev_input.borrow();
        let prev_input = prev_input.deref();

        let self_neu_grad = self.lr_params.get_2d_buf_t(TypeBuffer::NeuGrad);
        let mut self_neu_grad = self_neu_grad.borrow_mut();
        let self_neu_grad = self_neu_grad.deref_mut();

        let self_output = self.lr_params.get_2d_buf_t(TypeBuffer::Output);
        let self_output = self_output.borrow();
        let self_output = self_output.deref();

        let self_bias_grad = self
            .lr_params
            .get_1d_buf_t(TypeBuffer::BiasGrad);
        let mut self_bias_grad = self_bias_grad.borrow_mut();
        let self_bias_grad = self_bias_grad.deref_mut();

        // for each batch
        Zip::from(self_neu_grad.rows_mut())
            .and(expected_vec.rows())
            .and(self_output.rows()) 
            .par_for_each(|err_val_r, expected_r, output_r| {
                Zip::from(err_val_r).and(expected_r).and(output_r).for_each(
                    |err_val, expected, output| {
                        *err_val = (expected - output) * (self.activation.func_deriv)(*output);
                    },
                );
            });

        let ws_grad = self
            .lr_params
            .get_2d_buf_t(TypeBuffer::WeightsGrad);
        let mut ws_grad = ws_grad.borrow_mut();
        let ws_grad = ws_grad.deref_mut();

        let ws = self
            .lr_params
            .get_2d_buf_t(TypeBuffer::Weights);
        let ws = ws.borrow();
        let ws = ws.deref();

        let self_bias = self
            .lr_params
            .get_1d_buf_t(TypeBuffer::Bias);
        let mut self_bias = self_bias.borrow_mut();
        let self_bias = self_bias.deref_mut();

        // This could be done with parallel iterator
        // But  parallel iterator will make value only with big arrays (a lot of ws, big batch size)
        for ((self_neu_idx, prev_neu_idx), val) in ws_grad.indexed_iter_mut() {
            let cur_ws_idx = [self_neu_idx, prev_neu_idx];

            let mut avg = 0.0;
            
            Zip::from(prev_input.column(prev_neu_idx))
                .and(self_neu_grad.column(self_neu_idx))
                .for_each(|prev_val, err_val| {
                    avg += prev_val * err_val;
                });

            avg = avg / prev_input.column(prev_neu_idx).len() as f32;

            let mut l2_penalty = 0.0;
            if self.l2_regul != 0.0 {
                l2_penalty = self.l2_regul * ws[cur_ws_idx];
            }

            let mut l1_penalty = 0.0;
            if self.l1_regul == 0.0 {
                l1_penalty = self.l1_regul * sign(ws[cur_ws_idx]);
            }

            *val = avg - l2_penalty - l1_penalty;
        }

        Zip::from(self_neu_grad.columns()).and(self_bias_grad).and(self_bias).for_each(
            |err_vals, bias_grad, bias| {
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
            }
        );

        debug!("[ok] ErrorLayer backward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn layer_type(&self) -> &str {
        "EuclideanLossLayer"
    }

    fn cpu_params(&self) -> Option<CpuParams> {
        Some(self.lr_params.clone())
    }

    fn set_cpu_params(&mut self, lp: CpuParams) {
        self.lr_params = lp;
    }

    /// Carefull this method overwrites weights and all other params
    fn set_input_shape(&mut self, sh: &[usize]) {
        self.lr_params = CpuParams::new_with_bias(self.size, sh[0]);
    }

    fn size(&self) -> usize {
        self.size
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = EuclideanLossLayer::new(self.size, self.activation.clone());
        copy_l.set_cpu_params(self.lr_params.copy());
        Box::new(copy_l)
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }
}

impl<T, TD> EuclideanLossLayer<T, TD>
where
    T: Fn(f32) -> f32 + Clone,
    TD: Fn(f32) -> f32 + Clone,
{
    pub fn new(size: usize, activation: Activation<T, TD>) -> Self {
        Self {
            size,
            lr_params: CpuParams::empty(),
            activation,
            l1_regul: 0.0,
            l2_regul: 0.0,
        }
    }

    pub fn new_box(size: usize, activation: Activation<T, TD>) -> Box<Self> {
        Box::new(EuclideanLossLayer::new(size, activation))
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

impl<T, TD> WithParams for EuclideanLossLayer<T, TD>
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
