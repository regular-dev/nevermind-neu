use crate::optimizers::*;

use std::collections::HashMap;
use crate::util::*;
use crate::cpu_params::*;

#[derive(Clone)]
pub struct OptimizerRMS {
    pub learn_rate: f32,
    pub alpha: f32,
    pub theta: f32,
    pub rms: HashMap<u64, HashMap<i32, VariantParam>>,
}

impl OptimizerRMS {
    pub fn new(learn_rate: f32, alpha: f32) -> Self {
        Self {
            learn_rate,
            alpha,
            theta: 1e-8,
            rms: HashMap::new()
        }
    }
}

impl Default for OptimizerRMS {
    fn default() -> Self {
        Self {
            learn_rate: 1e-2,
            alpha: 0.9,
            theta: 1e-8,
            rms: HashMap::new()
        }
    }
}

impl OptimizerRMS {
    fn optimize_layer(
        buf: &mut [f32],
        buf_grad: &[f32],
        rms: &mut [f32],
        learn_rate: &f32,
        alpha: &f32,
        theta: &f32,
    ) {
        for ((buf_v, buf_grad_v), rms_v) in buf.iter_mut().zip(buf_grad.iter()).zip(rms.iter_mut()) {
            if *buf_grad_v == 0.0 {
                continue;
            }

            *rms_v = *alpha * *rms_v + (1.0 - alpha) * buf_grad_v.powf(2.0);
            *buf_v += (learn_rate / (*rms_v + theta).sqrt()) * buf_grad_v;
        }
    }
}

impl Optimizer for OptimizerRMS {
    fn optimize_params(&mut self, lp: &mut CpuParams, opt_prms: TrainableBufsIds) {
        if !self.rms.contains_key(&lp.id) {
            self.rms.insert(lp.id, HashMap::new());
        }

        for (buf_id, buf_grad_id) in opt_prms.0.iter().zip(opt_prms.1.iter()) {
            let buf_grad = lp.get_param(*buf_grad_id);
            let rms_val = self.rms.get_mut(&lp.id).unwrap();

            if !rms_val.contains_key(buf_grad_id) {
                let zeroed_param = VariantParam::copy_zeroed_shape_from(&buf_grad);
                rms_val.insert(*buf_grad_id, zeroed_param);
            }

            let rms_m = rms_val.get_mut(buf_grad_id).unwrap();

            match rms_m {
                VariantParam::Array1(arr1) => {
                    let buf_grad_slice = buf_grad.get_arr_1d();
                    let buf_grad_slice = buf_grad_slice.borrow();
                    let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

                    let buf_slice = lp.get_1d_buf(*buf_id);
                    let mut buf_slice = buf_slice.borrow_mut();
                    let buf_slice = buf_slice.as_slice_mut().unwrap();

                    let rms_slice = arr1.as_slice_mut().unwrap();

                    OptimizerRMS::optimize_layer(
                        buf_slice,
                        buf_grad_slice,
                        rms_slice,
                        &self.learn_rate,
                        &self.alpha,
                        &self.theta,
                    );
                }
                VariantParam::Array2(arr2) => {
                    let buf_grad_slice = buf_grad.get_arr_2d();
                    let buf_grad_slice = buf_grad_slice.borrow();
                    let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

                    let buf_slice = lp.get_2d_buf(*buf_id);
                    let mut buf_slice = buf_slice.borrow_mut();
                    let buf_slice = buf_slice.as_slice_mut().unwrap();

                    let rms_slice = arr2.as_slice_mut().unwrap();

                    OptimizerRMS::optimize_layer(
                        buf_slice,
                        buf_grad_slice,
                        rms_slice,
                        &self.learn_rate,
                        &self.alpha,
                        &self.theta,
                    );
                }
            }
        }
    }
}

impl WithParams for OptimizerRMS {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg_params = HashMap::new();

        cfg_params.insert("type".to_string(), Variant::String("rmsprop".to_string()));
        cfg_params.insert("learning_rate".to_string(), Variant::Float(self.learn_rate));
        cfg_params.insert("alpha".to_string(), Variant::Float(self.alpha));
        cfg_params.insert("theta".to_string(), Variant::Float(self.theta));

        cfg_params
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if args.contains_key("learning_rate") {
            if let Variant::Float(v) = args.get("learning_rate").unwrap() {
                self.learn_rate = *v;
            }
        }

        if args.contains_key("alpha") {
            if let Variant::Float(v) = args.get("alpha").unwrap() {
                self.alpha = *v;
            }
        }

        if args.contains_key("theta") {
            if let Variant::Float(v) = args.get("theta").unwrap() {
                self.theta = *v;
            }
        }
    }
}