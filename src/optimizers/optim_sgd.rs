use crate::optimizers::*;

use uuid::Uuid;

use std::ops::Deref;

use std::collections::HashMap;
use crate::util::*;

pub struct OptimizerSGD {
    pub learn_rate: f32,
    pub momentum: f32,
    pub ws_delta: HashMap<Uuid, WsBlob>,
}

impl OptimizerSGD {
    pub fn new(learn_rate: f32, momentum: f32) -> Self {
        Self {
            learn_rate,
            momentum,
            ws_delta: HashMap::new()
        }
    }

    fn optimize_layer_sgd(
        ws: &mut WsMat,
        ws_grad: &WsMat,
        ws_delta: &mut WsMat,
        learn_rate: &f32,
        momentum: &f32,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];

                // grad is 0.0 when weights is in dropout selection
                if ws_grad[cur_ws_idx] == 0.0 {
                    continue;
                }

                ws[cur_ws_idx] += momentum * ws_delta[cur_ws_idx];
                ws_delta[cur_ws_idx] = learn_rate * ws_grad[cur_ws_idx];
                ws[cur_ws_idx] += ws_delta[cur_ws_idx];
            }
        }
    }
}

impl Default for OptimizerSGD {
    fn default() -> Self {
        Self {
            learn_rate: 1e-2,
            momentum: 0.8,
            ws_delta: HashMap::new()
        }
    }
}

impl Optimizer for OptimizerSGD {
    fn optimize_network(&mut self, lp: &mut LearnParams) {
        let mut lr_ws = lp.ws.borrow_mut();
        let lr_grad = lp.ws_grad.borrow();

        if !self.ws_delta.contains_key(&lp.uuid) {
            let mut vec_init = Vec::new();
            for i in lr_ws.deref() {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }
            self.ws_delta.insert(lp.uuid, vec_init.clone());
        }

        let ws_delta = self.ws_delta.get_mut(&lp.uuid).unwrap();

        for (ws_idx, ws_iter) in lr_ws.iter_mut().enumerate() {
            OptimizerSGD::optimize_layer_sgd(
                ws_iter,
                &lr_grad[ws_idx],
                &mut ws_delta[ws_idx],
                &self.learn_rate,
                &self.momentum,
            );
        }
    }
    
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg_params = HashMap::new();

        cfg_params.insert("type".to_string(), Variant::String("sgd".to_string()));
        cfg_params.insert("learn_rate".to_string(), Variant::Float(self.learn_rate));
        cfg_params.insert("momentum".to_string(), Variant::Float(self.momentum));

        cfg_params
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if args.contains_key("learn_rate") {
            if let Variant::Float(v) = args.get("learn_rate").unwrap() {
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