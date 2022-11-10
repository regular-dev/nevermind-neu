use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::prelude::*;

use std::ops::Deref;

use log::debug;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use prost::Message;

use crate::layers_storage::*;
use crate::util::*;
use crate::solvers::*;
use crate::learn_params::*;
use crate::solvers::pb::PbSolverRms;

use uuid::Uuid;

#[derive(Default, Clone)]
pub struct SolverRMS {
    pub learn_rate: f32,
    pub momentum: f32,
    pub alpha: f32,
    pub theta: f32,
    layers: LayersStorage,
    batch_size: usize,
    rms: HashMap<Uuid, WsBlob>,
}

impl SolverRMS {
    pub fn new() -> Self {
        SolverRMS {
            learn_rate: 0.2,
            momentum: 0.2,
            alpha: 0.9,
            batch_size: 1,
            theta: 0.00000001,
            layers: LayersStorage::empty(),
            rms: HashMap::new(),
        }
    }

    pub fn batch(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self.layers.fit_to_batch_size(batch_size);
        self
    }

    fn optimizer_functor(
        lr: &mut LearnParams,
        rms: &mut HashMap<Uuid, WsBlob>,
        learn_rate: &f32,
        alpha: &f32,
        theta: &f32,
    ) {
        let lr_grad = lr.ws_grad.borrow();
        let mut lr_ws = lr.ws.borrow_mut();

        if !rms.contains_key(&lr.uuid) {
            let mut vec_init = Vec::new();
            for i in lr_ws.deref() {
                let ws_init = WsMat::zeros(i.raw_dim());
                vec_init.push(ws_init);
            }

            rms.insert(lr.uuid, vec_init);
        }

        let rms_mat = rms.get_mut(&lr.uuid).unwrap();

        for (ws_idx, ws_iter) in lr_ws.iter_mut().enumerate() {
            SolverRMS::optimize_layer_rms(
                ws_iter,
                &lr_grad[ws_idx],
                &mut rms_mat[ws_idx],
                learn_rate,
                alpha,
                theta,
            );
        }
    }

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

    pub fn from_file(filepath: &str) -> Result<Self, Box<dyn Error>> {
        let net_cfg_file = File::open(filepath)?;
        let solver_rms: SolverRMS = serde_yaml::from_reader(net_cfg_file)?;
        Ok(solver_rms)
    }
}

impl Solver for SolverRMS {
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

            SolverRMS::optimizer_functor(
                &mut lr_params,
                &mut self.rms,
                &self.learn_rate,
                &self.alpha,
                &self.theta,
            );
        }
    }

    fn solver_type(&self) -> &str {
        "rmsprop"
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

        let pb_solver = PbSolverRms {
            learn_rate: self.learn_rate,
            momentum: self.momentum,
            alpha: self.alpha,
            theta: self.theta,
            batch_cnt: self.batch_size as u32,
            rms: solver_helper::convert_hash_ws_blob_to_pb(&self.rms),
            layers: vec_lr,

        };

        // encode
        let mut file = File::create(filepath)?;

        file.write_all(pb_solver.encode_to_vec().as_slice())?;

        Ok(())
    }

    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>> {
        let buf = fs::read(filepath)?;

        let mut solver_rms = PbSolverRms::decode(buf.as_slice())?;

        self.learn_rate = solver_rms.learn_rate;
        self.momentum = solver_rms.momentum;
        self.alpha = solver_rms.alpha;
        self.theta = solver_rms.theta;
        self.batch_size = solver_rms.batch_cnt as usize;
        self.rms = solver_helper::convert_pb_to_hash_ws_blob(&mut solver_rms.rms);

        for (self_l, l) in self.layers.iter_mut().zip(&mut solver_rms.layers) {
            let layer_param = self_l.learn_params().unwrap();
            let mut l_ws = layer_param.ws.borrow_mut();
            *l_ws = solver_helper::convert_pb_to_ws_blob(l);
        }

        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct SerdeSolverRMS {
    pub learn_rate: f32,
    pub momentum: f32,
    pub alpha: f32,
    pub theta: f32,
    pub batch_size: usize,
    pub layers_cfg: LayersStorage,
}

impl Serialize for SolverRMS {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut solver_cfg = serializer.serialize_struct("SolverRMS Configuration", 3)?;
        solver_cfg.serialize_field("learn_rate", &self.learn_rate)?;
        solver_cfg.serialize_field("momentum", &self.momentum)?;
        solver_cfg.serialize_field("alpha", &self.alpha)?;
        solver_cfg.serialize_field("theta", &self.theta)?;
        solver_cfg.serialize_field("batch_size", &self.batch_size)?;
        solver_cfg.serialize_field("layers_cfg", &self.layers)?;
        solver_cfg.serialize_field("solver_type", self.solver_type())?;
        solver_cfg.end()
    }
}

impl<'de> Deserialize<'de> for SolverRMS {
    fn deserialize<D>(deserializer: D) -> Result<SolverRMS, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut s_solver = SerdeSolverRMS::deserialize(deserializer)?;

        let layer_storage = std::mem::replace(&mut s_solver.layers_cfg, LayersStorage::empty());

        let mut rms_solver = SolverRMS::new();
        rms_solver.learn_rate = s_solver.learn_rate;
        rms_solver.momentum = s_solver.momentum;
        rms_solver.alpha = s_solver.alpha;
        rms_solver.theta = s_solver.theta;
        rms_solver.layers = layer_storage;
        rms_solver = rms_solver.batch(s_solver.batch_size);

        Ok(rms_solver)
    }
}
