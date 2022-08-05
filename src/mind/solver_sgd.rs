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
use super::util::{Blob, Num, WsBlob, WsMat};

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
                let result_out = l.forward(vec![&input_data]);

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

            let result_out = l.forward(vec![out.unwrap()]);

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
                let result_out = l.backward(vec![expected_data], &WsBlob::new());

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

            let result_out = l.backward(vec![out.unwrap().0], out.unwrap().1);

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

            l.optimize(
                &mut |lr: &mut LearnParams, prev_lr: Option<&LearnParams>| {
                    // prev_lr could be empty for bias
                    let prev_vec = |idx: usize| -> Num {
                        match prev_lr {
                            Some(data) => {
                                return data.output[idx];
                            }
                            None => {
                                return 1.0;
                            }
                        }
                    };

                    if !self.ws_delta.contains_key(&lr.uuid) {
                        let ws_delta = WsMat::zeros(lr.ws[0].raw_dim());
                        self.ws_delta.insert(lr.uuid, vec![ws_delta]);
                    }

                    let ws_delta = self.ws_delta.get_mut(&lr.uuid).unwrap();

                    for neu_idx in 0..lr.ws[0].shape()[0] {
                        for prev_idx in 0..lr.ws[0].shape()[1] {
                            let cur_ws_idx = [neu_idx, prev_idx];
                            // ALPHA
                            lr.ws[0][cur_ws_idx] += self.alpha * ws_delta[0][cur_ws_idx];
                            // LEARNING RATE
                            ws_delta[0][cur_ws_idx] =
                                self.learn_rate * lr.err_vals[neu_idx] * prev_vec(prev_idx);

                            lr.ws[0][cur_ws_idx] += ws_delta[0][cur_ws_idx];
                        }
                    }
                },
                prev_lr.unwrap(),
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
