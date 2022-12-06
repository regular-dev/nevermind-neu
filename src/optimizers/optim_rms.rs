use crate::optimizers::*;

use uuid::Uuid;

use std::ops::Deref;

use std::collections::HashMap;
use crate::util::*;

pub struct OptimizerRMS {
    pub learn_rate: f32,
    pub alpha: f32,
    pub theta: f32,
    pub rms: HashMap<Uuid, WsBlob>,
}

impl OptimizerRMS {
    pub fn new(learn_rate: f32, alpha: f32) -> Self {
        Self {
            learn_rate,
            alpha,
            theta: 1e-6,
            rms: HashMap::new()
        }
    }
}

impl Default for OptimizerRMS {
    fn default() -> Self {
        Self {
            learn_rate: 1e-2,
            alpha: 0.8,
            theta: 1e-6,
            rms: HashMap::new()
        }
    }
}

impl OptimizerRMS {
    fn optimize_layer_rms(
        ws: &mut WsMat,
        ws_grad: &WsMat,
        rms: &mut WsMat,
        learn_rate: &f32,
        alpha: &f32,
        theta: &f32,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                rms[cur_ws_idx] =
                    alpha * rms[cur_ws_idx] + (1.0 - alpha) * ws_grad[cur_ws_idx].powf(2.0);
                ws[cur_ws_idx] +=
                    (learn_rate / (rms[cur_ws_idx] + theta).sqrt()) * ws_grad[cur_ws_idx];
            }
        }
    }
}

impl Optimizer for OptimizerRMS {
    fn optimize_network(&mut self, lp: &mut LearnParams) {
        let mut lr_ws = lp.ws.borrow_mut();
        let lr_grad = lp.ws_grad.borrow();

        if !self.rms.contains_key(&lp.uuid) {
            let mut vec_init = Vec::new();
            for i in lr_ws.deref() {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }
            self.rms.insert(lp.uuid, vec_init.clone());
        }

        let rms = self.rms.get_mut(&lp.uuid).unwrap();

        for (ws_idx, ws_iter) in lr_ws.iter_mut().enumerate() {
            OptimizerRMS::optimize_layer_rms(
                ws_iter,
                &lr_grad[ws_idx],
                &mut rms[ws_idx],
                &self.learn_rate,
                &self.alpha,
                &self.theta
            );
        }
    }
    
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg_params = HashMap::new();

        cfg_params.insert("type".to_string(), Variant::String("rmsprop".to_string()));
        cfg_params.insert("learn_rate".to_string(), Variant::Float(self.learn_rate));
        cfg_params.insert("alpha".to_string(), Variant::Float(self.alpha));
        cfg_params.insert("theta".to_string(), Variant::Float(self.theta));

        cfg_params
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if args.contains_key("learn_rate") {
            if let Variant::Float(v) = args.get("learn_rate").unwrap() {
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