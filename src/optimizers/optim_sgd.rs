use crate::optimizers::*;
use crate::cpu_params::*;
use crate::util::*;

use std::collections::HashMap;

pub struct OptimizerSGD {
    pub learn_rate: f32,
    pub momentum: f32,
    pub delta: HashMap<u64, HashMap<i32, VariantParam>>,
}

impl OptimizerSGD {
    pub fn new(learn_rate: f32, momentum: f32) -> Self {
        Self {
            learn_rate,
            momentum,
            delta: HashMap::new(),
        }
    }
}

impl Default for OptimizerSGD {
    fn default() -> Self {
        Self {
            learn_rate: 1e-2,
            momentum: 0.8,
            delta: HashMap::new(),
        }
    }
}

impl OptimizerSGD {
    fn optimize_layer(
        buf: &mut [f32],
        buf_grad: &[f32],
        delta: &mut [f32],
        learn_rate: &f32,
        momentum: &f32,
    ) {
        for ((buf_v, buf_grad_v), delta_v) in buf.iter_mut().zip(buf_grad.iter()).zip(delta.iter_mut()) {
            if *buf_grad_v == 0.0 {
                continue;
            }

            *buf_v += *momentum * *delta_v;
            *delta_v = *learn_rate * buf_grad_v;
            *buf_v += *delta_v;
        }
    }
}

impl Optimizer for OptimizerSGD {
    fn optimize_params(&mut self, lp: &mut CpuParams, opt_prms: TrainableBufsIds) {
        if !self.delta.contains_key(&lp.id) {
            self.delta.insert(lp.id, HashMap::new());
        }

        let delta_val = self.delta.get_mut(&lp.id).unwrap();

        for (buf_id, buf_grad_id) in opt_prms.0.iter().zip(opt_prms.1.iter()) {
            let buf_grad = lp.get_param(*buf_grad_id);

            if !delta_val.contains_key(buf_grad_id) {
                let zeroed_param = VariantParam::copy_zeroed_shape_from(&buf_grad);
                delta_val.insert(*buf_grad_id, zeroed_param);
            }

            let delta_m = delta_val.get_mut(buf_grad_id).unwrap();

            match delta_m {
                VariantParam::Array1(arr1) => {
                    let buf_grad_slice = buf_grad.get_arr_1d();
                    let buf_grad_slice = buf_grad_slice.borrow();
                    let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

                    let buf_slice = lp.get_1d_buf(*buf_id);
                    let mut buf_slice = buf_slice.borrow_mut();
                    let buf_slice = buf_slice.as_slice_mut().unwrap();

                    let delta_slice = arr1.as_slice_mut().unwrap();
                    OptimizerSGD::optimize_layer(
                        buf_slice,
                        buf_grad_slice,
                        delta_slice,
                        &self.learn_rate,
                        &self.momentum
                    );
                }
                VariantParam::Array2(arr2) => {
                    let buf_grad_slice = buf_grad.get_arr_2d();
                    let buf_grad_slice = buf_grad_slice.borrow();
                    let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

                    let buf_slice = lp.get_2d_buf(*buf_id);
                    let mut buf_slice = buf_slice.borrow_mut();
                    let buf_slice = buf_slice.as_slice_mut().unwrap();

                    let delta_slice = arr2.as_slice_mut().unwrap();
                    OptimizerSGD::optimize_layer(
                        buf_slice,
                        buf_grad_slice,
                        delta_slice,
                        &self.learn_rate,
                        &self.momentum
                    );
                }
            }
        }
    }
}

impl WithParams for OptimizerSGD {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg_params = HashMap::new();

        cfg_params.insert("type".to_string(), Variant::String("sgd".to_string()));
        cfg_params.insert("learning_rate".to_string(), Variant::Float(self.learn_rate));
        cfg_params.insert("momentum".to_string(), Variant::Float(self.momentum));

        cfg_params
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if args.contains_key("learning_rate") {
            if let Variant::Float(v) = args.get("learning_rate").unwrap() {
                self.learn_rate = *v;
            }
        }

        if args.contains_key("momentum") {
            if let Variant::Float(v) = args.get("momentum").unwrap() {
                self.momentum = *v;
            }
        }
    }
}
