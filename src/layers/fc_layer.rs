use std::collections::HashMap;

use ndarray::Zip;

use log::debug;

use rand::{thread_rng, Rng, ThreadRng};

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};
use crate::util::*;
use crate::learn_params::{LearnParams, ParamsBlob};

use crate::util::{Variant, WithParams};

// Fully-connected layer
#[derive(Clone)]
pub struct FcLayer<T: Fn(f32) -> f32 + Clone, TD: Fn(f32) -> f32 + Clone> {
    pub lr_params: LearnParams,
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
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws = self.lr_params.ws.borrow();
        let ws0 = &ws[0];
        let bias_out = ws[1].column(0);

        let dropout_len = (self.size as f32 * self.dropout) as usize;
        let dropout_n = self.rng.gen_range(0, self.size - dropout_len as usize);
        let dropout_y = dropout_n + dropout_len;

        // for each input batch
        Zip::from(inp_m.rows())
            .and(out_m.rows_mut())
            .par_for_each(|inp_b, out_b| {
                // for each batch
                let mul_res = ws0.dot(&inp_b);

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
        let next_err_vals = next_input[0].neu_grad.borrow();
        let next_ws = next_input[0].ws.borrow();
        let mut self_err_vals = self.lr_params.neu_grad.borrow_mut();
        let self_output = self.lr_params.output.borrow();
        let next_ws0 = &next_ws[0];

        Zip::from(self_err_vals.rows_mut())
            .and(next_err_vals.rows())
            .and(self_output.rows())
            .par_for_each(|err_val_r, next_err_val_r, output_r| {
                let mul_res = next_ws0.t().dot(&next_err_val_r);

                Zip::from(err_val_r).and(output_r).and(&mul_res).for_each(
                    |err_val, output, col| {
                        *err_val = (self.activation.func_deriv)(*output) * col;
                    },
                );
            });

        debug!("[hidden layer] i am here 2");

        // calc per-weight gradient, TODO : refactor code below
        // for prev_layer :
        let prev_input = prev_input[0].output.borrow();
        let ws = self.lr_params.ws.borrow();
        let mut ws_grad = self.lr_params.ws_grad.borrow_mut();

        // This could be done with parallel iterator
        // But parallel iterator will make value only with big arrays (a lot of ws, big batch size)
        for w in ws_grad[0].indexed_iter_mut() {
            let cur_ws_idx = [w.0.0, w.0.1];

            let mut avg = 0.0;
            Zip::from(prev_input.column(w.0.1)).and(self_err_vals.column(w.0.0)).for_each(
                |prev_val, err_val| {
                    avg += prev_val * err_val;
                }
            );

            avg = avg / prev_input.column(w.0.1).len() as f32;

            let mut l2_penalty = 0.0;
                if self.l2_regul != 0.0 {
                    l2_penalty = self.l2_regul * ws[0][cur_ws_idx];
                }

                let mut l1_penalty = 0.0;
                if self.l1_regul == 0.0 {
                    l1_penalty = self.l1_regul * sign(ws[0][cur_ws_idx]);
                }
            *w.1 = avg - l2_penalty - l1_penalty;
        }

        // bias
        for w in ws_grad[1].indexed_iter_mut() {
            let cur_ws_idx = [w.0.0, w.0.1];

                let mut avg = 0.0;
                Zip::from(self_err_vals.column(w.0.0)).for_each(|err_val| {
                    avg += err_val;
                });

                avg = avg / self_err_vals.column(w.0.1).len() as f32;

                let mut l2_penalty = 0.0;
                if self.l2_regul != 0.0 {
                    l2_penalty = self.l2_regul * ws[1][cur_ws_idx];
                }

                let mut l1_penalty = 0.0;
                if self.l1_regul == 0.0 {
                    l1_penalty = self.l1_regul * sign(ws[1][cur_ws_idx]);
                }

                *w.1 = avg - l2_penalty - l1_penalty;
        }

        debug!("[ok] HiddenLayer backward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.lr_params.clone())
    }

    fn set_learn_params(&mut self, lp: LearnParams) {
        self.lr_params = lp;
    }

    fn layer_type(&self) -> &str {
        "FcLayer"
    }

    /// Carefull this method overwrites weights and all other params
    fn set_input_shape(&mut self, sh: &[usize]) {
        self.lr_params = LearnParams::new_with_bias(self.size, sh[0]);
    }

    fn size(&self) -> usize {
        self.size
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = Box::new(FcLayer::new(self.size, self.activation.clone()));
        copy_l.set_learn_params(self.lr_params.copy());
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
            lr_params: LearnParams::empty(),
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
            self.lr_params = LearnParams::empty();
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
