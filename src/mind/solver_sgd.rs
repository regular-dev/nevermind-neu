use std::collections::HashMap;

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use log::debug;
use uuid::Uuid;

use ndarray::{Array, Array2}; // test purpose @xion

use super::abstract_layer::AbstractLayer;
use super::dataset::DataBatch;
use super::error_layer::ErrorLayer;
use super::hidden_layer::HiddenLayer;
use super::input_data_layer::InputDataLayer;
use super::layers_storage::LayersStorage;
use super::learn_params::LearnParams;
use super::solver::Solver;
use super::util::{Blob, DataVec, Num, WsBlob, WsMat};

// Train/Test Impl
pub struct SolverSGD {
    learn_rate: f32,
    alpha: f32,
    layers: LayersStorage,
    ws_delta: HashMap<Uuid, WsBlob>,
}

impl SolverSGD {
    pub fn new() -> Self {
        SolverSGD {
            layers: LayersStorage::new(),
            learn_rate: 0.2,
            alpha: 0.2,
            ws_delta: HashMap::new(),
        }
    }

    fn optimizer_functor(
        lr: &mut LearnParams,
        prev_lr: Vec<&mut LearnParams>,
        ws_delta: &mut HashMap<Uuid, WsBlob>,
        learn_rate: &f32,
        alpha: &f32,
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
            ws_delta.insert(lr.uuid, vec_init);
        }

        let ws_delta = ws_delta.get_mut(&lr.uuid).unwrap();

        for (ws_idx, ws_iter) in lr.ws.iter_mut().enumerate() {
            SolverSGD::optimize_layer_sgd( ws_iter, &mut ws_delta[ws_idx], learn_rate, alpha, &lr.err_vals, ws_idx, &mut prev_vec);
        }
    }

    fn optimize_layer_sgd(
        ws: &mut WsMat,
        ws_delta: &mut WsMat,
        learn_rate: &f32,
        alpha: &f32,
        err_vals: &DataVec,
        idx_ws: usize,
        fn_prev: &mut dyn FnMut(usize, usize) -> Num,
    ) {
        for neu_idx in 0..ws.shape()[0] {
            for prev_idx in 0..ws.shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                // ALPHA
                ws[cur_ws_idx] += alpha * ws_delta[cur_ws_idx];
                // LEARNING RATE
                ws_delta[cur_ws_idx] = learn_rate * err_vals[neu_idx] * fn_prev(idx_ws, prev_idx);

                ws[cur_ws_idx] += ws_delta[cur_ws_idx];
            }
        }
    }
}

impl Solver for SolverSGD {
    fn setup_network(&mut self, layers: LayersStorage) {
        self.layers = layers;
    }

    fn perform_step(&mut self, data: &DataBatch) {
        self.feedforward(&data, false);
        self.backpropagate(&data);
        self.optimize_network();
    }

    fn feedforward(&mut self, train_data: &DataBatch, print_out: bool) {
        let input_data = &train_data.input;

        let mut out = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            // handle input layer
            if idx == 0 {
                let result_out = l.forward(&vec![input_data]);

                match result_out {
                    Err(_reason) => {
                        return;
                    }
                    Ok(val) => {
                        out = Some(val);
                    }
                };
                continue;
            }

            let result_out = l.forward(&vec![out.unwrap()]);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            };
        }

        let out_val = &out.unwrap();

        if print_out {
            for i in out_val.iter() {
                println!("out val : {}", i);
            }
        }
    }

    fn backpropagate(&mut self, train_data: &DataBatch) {
        let expected_data = &train_data.expected;

        let mut out = None;

        for (idx, l) in self.layers.iter_mut().rev().enumerate() {
            if idx == 0 {
                let result_out = l.backward(&vec![expected_data], &WsBlob::new());

                match result_out {
                    Err(reason) => {
                        return;
                    }
                    Ok(val) => {
                        out = Some(val);
                    }
                }
                continue;
            }

            let result_out = l.backward(&vec![out.unwrap().0], out.unwrap().1);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            }
        }
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

            SolverSGD::optimizer_functor(
                l.learn_params().unwrap(),
                prev_lr_vec,
                &mut self.ws_delta,
                &self.learn_rate,
                &self.learn_rate,
            );

            prev_lr = l.learn_params();
        }
    }
}

impl Serialize for SolverSGD {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut solver_cfg = serializer.serialize_struct("Solver Configuration", 3)?;
        solver_cfg.serialize_field("learning_rate", &self.learn_rate)?;
        solver_cfg.serialize_field("alpha", &self.alpha)?;
        solver_cfg.serialize_field("layers_cfg", &self.layers)?;
        solver_cfg.end()
    }
}
