use std::collections::HashMap;

use log::debug;

use serde::ser::SerializeStruct;
use serde::ser::{Serialize, Serializer};

use super::dataset::DataBatch;
use super::layers_storage::LayersStorage;
use super::learn_params::LearnParams;
use super::solver::Solver;
use super::solver_helper;
use super::util::{Num, WsBlob, WsMat, DataVec};
use uuid::Uuid;

pub struct SolverRMS {
    learn_rate: f32,
    momentum: f32,
    alpha: f32,
    theta: f32,
    layers: LayersStorage,
    ws_delta: HashMap<Uuid, WsBlob>,
    err_rms: HashMap<Uuid, DataVec>
}

impl SolverRMS {
    pub fn new() -> Self {
        SolverRMS {
            learn_rate: 0.02,
            momentum: 0.2,
            alpha: 0.9,
            theta: 0.00000001,
            layers: LayersStorage::new(),
            ws_delta: HashMap::new(),
            err_rms: HashMap::new()
        }
    }

    fn optimizer_functor(
        lr: &mut LearnParams,
        prev_lr: Vec<&mut LearnParams>,
        ws_delta: &mut HashMap<Uuid, WsBlob>,
        err_rms: &mut HashMap<Uuid, DataVec>,
        learn_rate: &f32,
        momentum: &f32,
        alpha: &f32,
        theta: &f32
    ) {
        // prev_lr could be empty for bias

        let mut prev_vec = |idx_vec: usize, idx_err: usize| -> Num {
            if let Some(prev_lr_vec) = prev_lr.get(idx_vec) {
                if let Some(prev_lr_val) = prev_lr_vec.output.get(idx_err) {
                    return *prev_lr_val;
                }
            } else {
                return 1.0;
            }

            return 1.0;
        };

        if !ws_delta.contains_key(&lr.uuid) {
            let mut vec_init = Vec::new();
            for i in &lr.ws {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }

            let err_rms_dv = DataVec::zeros(lr.err_vals.len());

            ws_delta.insert(lr.uuid, vec_init);
            err_rms.insert(lr.uuid, err_rms_dv);
        }

        let ws_delta = ws_delta.get_mut(&lr.uuid).unwrap();
        let err_rms = err_rms.get_mut(&lr.uuid).unwrap();

        for (ws_idx, ws_iter) in lr.ws.iter_mut().enumerate() {
            SolverRMS::optimize_layer_sgd(
                ws_iter,
                &mut ws_delta[ws_idx],
                err_rms,
                learn_rate,
                momentum,
                alpha,
                theta,
                &lr.err_vals,
                ws_idx,
                &mut prev_vec,
            );
        }
    }

    fn optimize_layer_sgd(
        ws: &mut WsMat,
        ws_delta: &mut WsMat,
        err_rms: &mut DataVec,
        learn_rate: &f32,
        momentum: &f32,
        alpha: &f32,
        theta: &f32,
        err_vals: &DataVec,
        idx_ws: usize,
        fn_prev: &mut dyn FnMut(usize, usize) -> Num,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                let grad = err_vals[neu_idx] * fn_prev(idx_ws, prev_idx);

                ws_delta[cur_ws_idx] = alpha * ws_delta[cur_ws_idx] +
                                       (1.0 - alpha) * grad.powf(2.0);
                ws[cur_ws_idx] += ( learn_rate / (ws_delta[cur_ws_idx] + theta).sqrt() ) *
                                   grad;
            }
        }
    }
}

impl Solver for SolverRMS {
    fn setup_network(&mut self, layers: LayersStorage) {
        self.layers = layers;
    }

    fn perform_step(&mut self, data: &DataBatch) {
        self.feedforward(&data, false);
        self.backpropagate(&data);
        self.optimize_network();
    }

    fn feedforward(&mut self, train_data: &DataBatch, print_out: bool) {
        solver_helper::feedforward(&mut self.layers, train_data, print_out);
    }

    fn backpropagate(&mut self, train_data: &DataBatch) {
        solver_helper::backpropagate(&mut self.layers, train_data);
    }

    fn optimize_network(&mut self) {
        let mut prev_lr = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            if idx == 0 {
                prev_lr = l.learn_params();
                continue;
            }

            debug!("current optimizing layer : {}", l.layer_type());

            let mut prev_lr_vec = Vec::new();

            match prev_lr {
                Some(val) => prev_lr_vec.push(val),
                None => (),
            }

            SolverRMS::optimizer_functor(
                l.learn_params().unwrap(),
                prev_lr_vec,
                &mut self.ws_delta,
                &mut self.err_rms,
                &self.learn_rate,
                &self.momentum,
                &self.alpha,
                &self.theta
            );

            prev_lr = l.learn_params();
        }
    }
}

impl Serialize for SolverRMS {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut solver_cfg = serializer.serialize_struct("SolverRMS Configuration", 3)?;
        solver_cfg.serialize_field("learning_rate", &self.learn_rate)?;
        solver_cfg.serialize_field("alpha", &self.momentum)?;
        solver_cfg.serialize_field("layers_cfg", &self.layers)?;
        solver_cfg.end()
    }
}
