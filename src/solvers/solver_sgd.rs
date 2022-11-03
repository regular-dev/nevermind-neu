use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::ops::Deref;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use log::debug;
use uuid::Uuid;

use prost::Message;

use super::solver::{pb::PbSolverSgd, Solver};
use super::solver_helper;
use crate::layers_storage::LayersStorage;
use crate::learn_params::LearnParams;
use crate::util::{Batch, WsBlob, WsMat};

// Train/Test Impl
pub struct SolverSGD {
    pub learn_rate: f32,
    pub momentum: f32,
    layers: LayersStorage,
    ws_delta: HashMap<Uuid, WsBlob>,
    batch_size: usize,
}

impl SolverSGD {
    pub fn new() -> Self {
        SolverSGD {
            layers: LayersStorage::new(),
            learn_rate: 0.05,
            momentum: 0.2,
            ws_delta: HashMap::new(),
            batch_size: 1,
        }
    }

    fn optimizer_functor(
        lr: &mut LearnParams,
        ws_delta: &mut HashMap<Uuid, WsBlob>,
        learn_rate: &f32,
        alpha: &f32,
    ) {
        let mut lr_ws = lr.ws.borrow_mut();
        let lr_grad = lr.ws_grad.borrow();

        if !ws_delta.contains_key(&lr.uuid) {
            let mut vec_init = Vec::new();
            for i in lr_ws.deref() {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }
            ws_delta.insert(lr.uuid, vec_init.clone());
        }

        let ws_delta = ws_delta.get_mut(&lr.uuid).unwrap();

        for (ws_idx, ws_iter) in lr_ws.iter_mut().enumerate() {
            SolverSGD::optimize_layer_sgd(
                ws_iter,
                &lr_grad[ws_idx],
                &mut ws_delta[ws_idx],
                learn_rate,
                alpha,
            );
        }
    }

    fn optimize_layer_sgd(
        ws: &mut WsMat,
        ws_grad: &WsMat,
        ws_delta: &mut WsMat,
        learn_rate: &f32,
        alpha: &f32,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                // ALPHA
                ws[cur_ws_idx] += alpha * ws_delta[cur_ws_idx];
                // LEARNING RATE
                ws_delta[cur_ws_idx] = learn_rate * ws_grad[cur_ws_idx];

                ws[cur_ws_idx] += ws_delta[cur_ws_idx];
            }
        }
    }

    pub fn batch(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self.layers.fit_to_batch_size(batch_size);
        self
    }

    pub fn from_file(filepath: &str) -> Result<Self, Box<dyn Error>> {
        let net_cfg_file = File::open(filepath)?;
        let solver_sgd: SolverSGD = serde_yaml::from_reader(net_cfg_file)?;
        Ok(solver_sgd)
    }
}

impl Solver for SolverSGD {
    fn setup_network(&mut self, layers: LayersStorage) {
        self.layers = layers;
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn feedforward(&mut self, train_data: Batch, print_out: bool) {
        solver_helper::feedforward(&mut self.layers, train_data, print_out);
    }

    fn backpropagate(&mut self, train_data: Batch) {
        solver_helper::backpropagate(&mut self.layers, train_data);
    }

    fn optimize_network(&mut self) {
        for (idx, l) in self.layers.iter_mut().enumerate() {
            if idx == 0 {
                continue;
            }

            debug!("current optimizing layer : {}", l.layer_type());

            let mut lr_params = l.learn_params().unwrap();

            SolverSGD::optimizer_functor(
                &mut lr_params,
                &mut self.ws_delta,
                &self.learn_rate,
                &self.momentum,
            );
        }
    }

    fn solver_type(&self) -> &str {
        "sgd"
    }

    fn layers(&self) -> &LayersStorage {
        &self.layers
    }

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>> {
        // create vector of layers learn_params
        let mut vec_lr = Vec::new();
        for l in 0..self.layers.len() {
            let lr_params = self.layers.at(l).learn_params().unwrap();
            let ws = lr_params.ws.borrow();
            vec_lr.push(solver_helper::convert_ws_blob_to_pb(ws.deref()));
        }

        let pb_solver = PbSolverSgd {
            learn_rate: self.learn_rate,
            momentum: self.momentum,
            batch_cnt: self.batch_size as u32,
            ws_delta: solver_helper::convert_hash_ws_blob_to_pb(&self.ws_delta),
            layers: vec_lr,
        };

        // encode
        let mut file = File::create(filepath)?;

        file.write_all(pb_solver.encode_to_vec().as_slice())?;

        Ok(())
    }

    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>> {
        let buf = fs::read(filepath)?;

        let mut solver_rms = PbSolverSgd::decode(buf.as_slice())?;

        self.learn_rate = solver_rms.learn_rate;
        self.momentum = solver_rms.momentum;
        self.batch_size = solver_rms.batch_cnt as usize;
        self.ws_delta = solver_helper::convert_pb_to_hash_ws_blob(&mut solver_rms.ws_delta);

        for (self_l, l) in self.layers.iter_mut().zip(&mut solver_rms.layers) {
            let layer_param = self_l.learn_params().unwrap();
            let mut l_ws = layer_param.ws.borrow_mut();
            *l_ws = solver_helper::convert_pb_to_ws_blob(l);
        }

        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct SerdeSolverSGD {
    pub learn_rate: f32,
    pub momentum: f32,
    pub batch_size: usize,
    pub layers_cfg: LayersStorage,
}

impl Serialize for SolverSGD {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut solver_cfg = serializer.serialize_struct("SolverSGD Configuration", 3)?;
        solver_cfg.serialize_field("learn_rate", &self.learn_rate)?;
        solver_cfg.serialize_field("momentum", &self.momentum)?;
        solver_cfg.serialize_field("batch_size", &self.batch_size)?;
        solver_cfg.serialize_field("layers_cfg", &self.layers)?;
        solver_cfg.serialize_field("solver_type", self.solver_type())?;

        solver_cfg.end()
    }
}

impl<'de> Deserialize<'de> for SolverSGD {
    fn deserialize<D>(deserializer: D) -> Result<SolverSGD, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut s_solver = SerdeSolverSGD::deserialize(deserializer)?;

        let layer_storage = std::mem::replace(&mut s_solver.layers_cfg, LayersStorage::new());

        let mut sgd_solver = SolverSGD::new();
        sgd_solver.learn_rate = s_solver.learn_rate;
        sgd_solver.momentum = s_solver.momentum;
        sgd_solver.layers = layer_storage;
        let sgd_solver = sgd_solver.batch(s_solver.batch_size);

        Ok(sgd_solver)
    }
}
