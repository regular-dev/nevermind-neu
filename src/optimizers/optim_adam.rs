use crate::optimizers::*;

use uuid::Uuid;

use std::ops::Deref;
use std::collections::HashMap;
use crate::util::*;

use log::info;

pub struct OptimizerAdam {
    pub learn_rate: f32,
    pub theta: f32,
    pub b1: f32,
    pub b2: f32,
    pub v: HashMap<Uuid, WsBlob>,
    pub m: HashMap<Uuid, WsBlob>,
}

impl OptimizerAdam {
    pub fn new(learn_rate: f32) -> Self {
        Self {
            learn_rate,
            theta: 1e-8,
            b1: 0.9,
            b2: 0.99,
            v: HashMap::new(),
            m: HashMap::new()
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
            m: HashMap::new()
        }
    }
}

impl OptimizerAdam {
    fn optimize_layer(
        ws: &mut WsMat,
        ws_grad: &WsMat,
        v: &mut WsMat,
        m: &mut WsMat,
        learn_rate: &f32,
        theta: &f32,
        b1: &f32,
        b2: &f32,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];

                // grad is 0.0 when weights is in dropout selection
                if f32::is_nan(ws_grad[cur_ws_idx]) {
                    continue;
                }

                m[cur_ws_idx] = b1 * m[cur_ws_idx] + (1.0 - b1) * ws_grad[cur_ws_idx];
                v[cur_ws_idx] = b2 * v[cur_ws_idx] + (1.0 - b2) * ws_grad[cur_ws_idx].powf(2.0);
                ws[cur_ws_idx] += learn_rate / (v[cur_ws_idx] + theta).sqrt() * m[cur_ws_idx];  
            }
        }
    }
}

impl Optimizer for OptimizerAdam {
    fn optimize_params(&mut self, lp: &mut LearnParams) {
        let mut lr_ws = lp.ws.borrow_mut();
        let lr_grad = lp.ws_grad.borrow();

        if !self.v.contains_key(&lp.uuid) {
            let mut vec_init = Vec::new();
            for i in lr_ws.deref() {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }
            self.v.insert(lp.uuid, vec_init.clone());
            self.m.insert(lp.uuid, vec_init);
        }

        let v = self.v.get_mut(&lp.uuid).unwrap();
        let m = self.m.get_mut(&lp.uuid).unwrap();

        for (ws_idx, ws_iter) in lr_ws.iter_mut().enumerate() {
            OptimizerAdam::optimize_layer(
                ws_iter,
                &lr_grad[ws_idx],
                &mut v[ws_idx],
                &mut m[ws_idx],
                &self.learn_rate,
                &self.theta,
                &self.b1,
                &self.b2,
            );
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