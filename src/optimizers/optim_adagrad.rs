use crate::optimizers::*;
use crate::cpu_params::*;
use crate::util::*;

use log::debug;

use std::collections::HashMap;

pub struct OptimizerAdaGrad {
    pub learn_rate: f32,
    pub theta: f32,
    pub g: HashMap<u64, HashMap<i32, VariantParam>>,
}

impl OptimizerAdaGrad {
    pub fn new(learn_rate: f32) -> Self {
        Self {
            learn_rate,
            theta: 1e-8,
            g: HashMap::new(),
        }
    }
}

impl Default for OptimizerAdaGrad {
    fn default() -> Self {
        Self {
            learn_rate: 1e-2,
            theta: 1e-6,
            g: HashMap::new(),
        }
    }
}

impl OptimizerAdaGrad {
    fn optimize_layer(
        buf: &mut [f32],
        buf_grad: &[f32],
        g: &mut [f32],
        learn_rate: &f32,
        theta: &f32,
    ) {
        for ((buf_v, buf_grad_v), g_v) in buf.iter_mut().zip(buf_grad.iter()).zip(g.iter_mut()) {
            if *buf_grad_v == 0.0 {
                continue;
            }

            *g_v += buf_grad_v.powf(2.0);
            *buf_v += (learn_rate / (*g_v + theta).sqrt()) * buf_grad_v;
        }
    }
}

impl Optimizer for OptimizerAdaGrad {
    fn optimize_params(&mut self, lp: &mut CpuParams, opt_prms: TrainableBufsIds) {
        if !self.g.contains_key(&lp.id) {
            self.g.insert(lp.id, HashMap::new());
            debug!("[opt_ada_grad] Inserted learn_params with id {}", lp.id);
        }

        for (buf_id, buf_grad_id) in opt_prms.0.iter().zip(opt_prms.1.iter()) {
            let buf_grad = lp.get_param(*buf_grad_id);
            let g_val = self.g.get_mut(&lp.id).unwrap();

            if !g_val.contains_key(buf_grad_id) {
                let zeroed_param = VariantParam::copy_zeroed_shape_from(&buf_grad);
                g_val.insert(*buf_grad_id, zeroed_param);
            }

            let g_m = g_val.get_mut(buf_grad_id).unwrap();

            match g_m {
                VariantParam::Array1(arr1) => {
                    let buf_grad_slice = buf_grad.get_arr_1d();
                    let buf_grad_slice = buf_grad_slice.borrow();
                    let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

                    let buf_slice = lp.get_1d_buf(*buf_id);
                    let mut buf_slice = buf_slice.borrow_mut();
                    let buf_slice = buf_slice.as_slice_mut().unwrap();

                    let v_slice = arr1.as_slice_mut().unwrap();

                    OptimizerAdaGrad::optimize_layer(
                        buf_slice,
                        buf_grad_slice,
                        v_slice,
                        &self.learn_rate,
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

                    let v_slice = arr2.as_slice_mut().unwrap();

                    OptimizerAdaGrad::optimize_layer(
                        buf_slice,
                        buf_grad_slice,
                        v_slice,
                        &self.learn_rate,
                        &self.theta,
                    );
                }
            }
        }

        // match g_m {
        //     VariantParam::Array1(arr1) => {
        //         let buf_grad_slice = lp.get_1d_buf(*buf_grad_id);
        //         let buf_grad_slice = buf_grad_slice.borrow();
        //         let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

        //         let buf_slice = lp.get_1d_buf(*buf_id);
        //         let mut buf_slice = buf_slice.borrow_mut();
        //         let buf_slice = buf_slice.as_slice_mut().unwrap();

        //         let g_slice = arr1.as_slice_mut().unwrap();

        //         OptimizerAdaGrad::optimize_layer(
        //             buf_slice,
        //             buf_grad_slice,
        //             g_slice,
        //             &self.learn_rate,
        //             &self.theta,
        //         );
        //     },
        //     VariantParam::Array2(mut arr2) => {
        //         let buf_grad_slice = lp.get_2d_buf(*buf_grad_id);
        //         let buf_grad_slice = buf_grad_slice.borrow();
        //         let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

        //         let mut buf_slice = lp.get_2d_buf(*buf_id);
        //         let mut buf_slice = buf_slice.borrow_mut();
        //         let mut buf_slice = buf_slice.as_slice_mut().unwrap();

        //         let g_slice = arr2.as_slice_mut().unwrap();

        //         OptimizerAdaGrad::optimize_layer(
        //             buf_slice,
        //             buf_grad_slice,
        //             g_slice,
        //             &self.learn_rate,
        //             &self.theta,
        //         );
        //     }
        // }
    }
}

impl WithParams for OptimizerAdaGrad {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg_params = HashMap::new();

        cfg_params.insert("type".to_string(), Variant::String("adagrad".to_string()));
        cfg_params.insert("learning_rate".to_string(), Variant::Float(self.learn_rate));
        cfg_params.insert("theta".to_string(), Variant::Float(self.theta));

        cfg_params
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if args.contains_key("learning_rate") {
            if let Variant::Float(v) = args.get("learning_rate").unwrap() {
                self.learn_rate = *v;
            }
        }

        if args.contains_key("theta") {
            if let Variant::Float(v) = args.get("theta").unwrap() {
                self.theta = *v;
            }
        }
    }
}
