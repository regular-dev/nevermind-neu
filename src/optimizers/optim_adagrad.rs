use crate::optimizers::*;

use uuid::Uuid;

use std::ops::Deref;

use std::collections::HashMap;
use crate::util::*;

pub struct OptimizerAdaGrad {
    pub learn_rate: f32,
    pub theta: f32,
    pub g: HashMap<Uuid, WsBlob>,
}

impl OptimizerAdaGrad {
    pub fn new(learn_rate: f32) -> Self {
        Self {
            learn_rate,
            theta: 1e-8,
            g: HashMap::new()
        }
    }
}

impl Default for OptimizerAdaGrad {
    fn default() -> Self {
        Self {
            learn_rate: 1e-2,
            theta: 1e-6,
            g: HashMap::new()
        }
    }
}

impl OptimizerAdaGrad {
    fn optimize_layer(
        ws: &mut WsMat,
        ws_grad: &WsMat,
        g: &mut WsMat,
        learn_rate: &f32,
        theta: &f32,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];

                // grad is 0.0 when weights is in dropout selection
                if ws_grad[cur_ws_idx] == 0.0 {
                    continue;
                }

                g[cur_ws_idx] += ws_grad[cur_ws_idx].powf(2.0);
                ws[cur_ws_idx] +=
                    (learn_rate / (g[cur_ws_idx] + theta).sqrt()) * ws_grad[cur_ws_idx];
            }
        }
    }
}

impl Optimizer for OptimizerAdaGrad {
    fn optimize_params(&mut self, lp: &mut LearnParams) {
        let mut lr_ws = lp.ws.borrow_mut();
        let lr_grad = lp.ws_grad.borrow();

        if !self.g.contains_key(&lp.uuid) {
            let mut vec_init = Vec::new();
            for i in lr_ws.deref() {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }
            self.g.insert(lp.uuid, vec_init.clone());
        }

        let g = self.g.get_mut(&lp.uuid).unwrap();

        for (ws_idx, ws_iter) in lr_ws.iter_mut().enumerate() {
            OptimizerAdaGrad::optimize_layer(
                ws_iter,
                &lr_grad[ws_idx],
                &mut g[ws_idx],
                &self.learn_rate,
                &self.theta
            );
        }
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