use std::collections::HashMap;
use std::ops::Deref;

use log::{debug, error};

use serde::ser::SerializeStruct;
use serde::ser::{Serialize, Serializer};

use super::dataset::DataBatch;
use super::layers_storage::LayersStorage;
use super::learn_params::LearnParams;
use super::solver::{Solver, BatchCounter};
use super::solver_helper;
use super::util::{Num, WsBlob, WsMat, DataVec};
use uuid::Uuid;

pub struct SolverRMS {
    learn_rate: f32,
    momentum: f32,
    alpha: f32,
    theta: f32,
    layers: LayersStorage,
    batch_cnt: BatchCounter,
    rms: HashMap<Uuid, WsBlob>,
    ws_batch: HashMap<Uuid, WsBlob>
}

impl SolverRMS {
    pub fn new() -> Self {
        SolverRMS {
            learn_rate: 0.02,
            momentum: 0.2,
            alpha: 0.9,
            batch_cnt: BatchCounter::new(1),
            theta: 0.00000001,
            layers: LayersStorage::new(),
            rms: HashMap::new(),
            ws_batch: HashMap::new()
        }
    }

    pub fn batch(mut self, batch_size: usize) -> Self {
        self.batch_cnt.batch_size = batch_size;
        self
    }

    fn optimizer_functor(
        lr: &mut LearnParams,
        rms: &mut HashMap<Uuid, WsBlob>,
        ws_batch: &mut HashMap<Uuid, WsBlob>,
        learn_rate: &f32,
        alpha: &f32,
        theta: &f32,
        update_ws: bool
    ) {
        if !rms.contains_key(&lr.uuid) {
            let mut vec_init = Vec::new();
            for i in lr.ws.borrow().deref() {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }

            ws_batch.insert(lr.uuid, vec_init.clone());
            rms.insert(lr.uuid, vec_init);
        }

        let rms_mat = rms.get_mut(&lr.uuid).unwrap();
        let batch_mat = ws_batch.get_mut(&lr.uuid).unwrap();

        for (ws_idx, ws_iter) in lr.ws.borrow_mut().iter_mut().enumerate() {
            SolverRMS::optimize_layer_rms(
                ws_iter,
                &lr.ws_grad.borrow()[ws_idx],
                &mut rms_mat[ws_idx],
                &mut batch_mat[ws_idx],
                learn_rate,
                alpha,
                theta,
                lr.err_vals.borrow().deref(),
                ws_idx,
                update_ws,
            );
        }
    }

    fn optimize_layer_rms(
        ws: &mut WsMat,
        ws_grad: &WsMat,
        rms: &mut WsMat,
        ws_batch: &mut WsMat,
        learn_rate: &f32,
        alpha: &f32,
        theta: &f32,
        err_vals: &DataVec,
        idx_ws: usize,
        is_upd: bool,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                // let grad = err_vals[neu_idx] * fn_prev(idx_ws, prev_idx);
                rms[cur_ws_idx] = alpha * rms[cur_ws_idx] +
                                    (1.0 - alpha) * ws_grad[cur_ws_idx].powf(2.0);
                ws_batch[cur_ws_idx] += ( learn_rate / (rms[cur_ws_idx] + theta).sqrt() ) *
                                    ws_grad[cur_ws_idx];

                if is_upd {
                    ws[cur_ws_idx] += ws_batch[cur_ws_idx];
                    ws_batch[cur_ws_idx] = 0.0;
                }
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
        let is_upd = self.batch_cnt.is_update();

        let mut prev_lr = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            if idx == 0 {
                prev_lr = l.learn_params();
                continue;
            }

            debug!("current optimizing layer : {}", l.layer_type());

            let mut lr_params = l.learn_params().unwrap();

            SolverRMS::optimizer_functor(
                &mut lr_params,
                &mut self.rms,
                &mut self.ws_batch,
                &self.learn_rate,
                &self.alpha,
                &self.theta,
                is_upd
            );

            prev_lr = l.learn_params();
        }

        self.batch_cnt.increment();
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
