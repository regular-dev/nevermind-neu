use crate::optimizers::*;
use crate::cpu_params::*;
use crate::util::*;

use std::collections::HashMap;

pub struct OptimizerAdam {
    pub learn_rate: f32,
    pub theta: f32,
    pub b1: f32,
    pub b2: f32,
    pub v: HashMap<u64, HashMap<i32, VariantParam>>,
    pub m: HashMap<u64, HashMap<i32, VariantParam>>,
}

impl OptimizerAdam {
    pub fn new(learn_rate: f32) -> Self {
        Self {
            learn_rate,
            theta: 1e-8,
            b1: 0.9,
            b2: 0.99,
            v: HashMap::new(),
            m: HashMap::new(),
        }
    }
}

impl Default for OptimizerAdam {
    fn default() -> Self {
        Self {
            learn_rate: 3e-4,
            b1: 0.9,
            b2: 0.99,
            theta: 1e-8,
            v: HashMap::new(),
            m: HashMap::new(),
        }
    }
}

// m[cur_ws_idx] = b1 * m[cur_ws_idx] + (1.0 - b1) * ws_grad[cur_ws_idx];
// v[cur_ws_idx] = b2 * v[cur_ws_idx] + (1.0 - b2) * ws_grad[cur_ws_idx].powf(2.0);
// ws[cur_ws_idx] += learn_rate / (v[cur_ws_idx] + theta).sqrt() * m[cur_ws_idx];

impl OptimizerAdam {
    fn optimize_layer(
        buf: &mut [f32],
        buf_grad: &[f32],
        v: &mut [f32],
        m: &mut [f32],
        learn_rate: &f32,
        theta: &f32,
        b1: &f32,
        b2: &f32,
    ) {
        for (((buf_v, buf_grad_v), v_v), m_v) in buf
            .iter_mut()
            .zip(buf_grad.iter())
            .zip(v.iter_mut())
            .zip(m.iter_mut())
        {
            if *buf_grad_v == 0.0 {
                continue;
            }

            *m_v = *b1 * *m_v + (1.0 - b1) * buf_grad_v;
            *v_v = *b2 * *v_v + (1.0 - b2) * buf_grad_v.powf(2.0);
            *buf_v += learn_rate / (*v_v + theta).sqrt() * *m_v;
        }
    }
}

impl Optimizer for OptimizerAdam {
    fn optimize_params(&mut self, lp: &mut CpuParams, opt_prms: TrainableBufsIds) {
        if !self.v.contains_key(&lp.id) {
            self.v.insert(lp.id, HashMap::new());
            self.m.insert(lp.id, HashMap::new());
        }

        for (buf_id, buf_grad_id) in opt_prms.0.iter().zip(opt_prms.1.iter()) {
            let buf_grad = lp.get_param(*buf_grad_id);
            let v = self.v.get_mut(&lp.id).unwrap();
            let m = self.m.get_mut(&lp.id).unwrap();

            if !v.contains_key(buf_grad_id) {
                let zeroed_param = VariantParam::copy_zeroed_shape_from(&buf_grad);
                v.insert(*buf_grad_id, zeroed_param);
            }

            if !m.contains_key(buf_grad_id) {
                let zeroed_param = VariantParam::copy_zeroed_shape_from(&buf_grad);
                m.insert(*buf_grad_id, zeroed_param);
            }

            let v_m = v.get_mut(buf_grad_id).unwrap();
            let m_m = m.get_mut(buf_grad_id).unwrap();

            match v_m {
                VariantParam::Array1(arr1) => {
                    let buf_grad_slice = buf_grad.get_arr_1d();
                    let buf_grad_slice = buf_grad_slice.borrow();
                    let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

                    let buf_slice = lp.get_1d_buf(*buf_id);
                    let mut buf_slice = buf_slice.borrow_mut();
                    let buf_slice = buf_slice.as_slice_mut().unwrap();

                    let v_slice = arr1.as_slice_mut().unwrap();

                    if let VariantParam::Array1(arr1_m) = m_m {
                        OptimizerAdam::optimize_layer(
                            buf_slice,
                            buf_grad_slice,
                            v_slice,
                            arr1_m.as_slice_mut().unwrap(),
                            &self.learn_rate,
                            &self.theta,
                            &self.b1,
                            &self.b2,
                        );
                    }
                },
                VariantParam::Array2(arr2) => {
                    let buf_grad_slice = buf_grad.get_arr_2d();
                    let buf_grad_slice = buf_grad_slice.borrow();
                    let buf_grad_slice = buf_grad_slice.as_slice().unwrap();

                    let buf_slice = lp.get_2d_buf(*buf_id);
                    let mut buf_slice = buf_slice.borrow_mut();
                    let buf_slice = buf_slice.as_slice_mut().unwrap();

                    let v_slice = arr2.as_slice_mut().unwrap();

                    if let VariantParam::Array2(arr2_m) = m_m {
                        OptimizerAdam::optimize_layer(
                            buf_slice,
                            buf_grad_slice,
                            v_slice,
                            arr2_m.as_slice_mut().unwrap(),
                            &self.learn_rate,
                            &self.theta,
                            &self.b1,
                            &self.b2,
                        );
                    }
                }
            }
        }
    }
}

impl WithParams for OptimizerAdam {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg_params = HashMap::new();

        cfg_params.insert("type".to_string(), Variant::String("adam".to_string()));
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
