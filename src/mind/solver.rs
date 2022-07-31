use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use log::debug;

use super::abstract_layer::AbstractLayer;
use super::dataset::DataBatch;
use super::error_layer::ErrorLayer;
use super::hidden_layer::HiddenLayer;
use super::input_data_layer::InputDataLayer;
use super::layers_storage::LayersStorage;
use super::learn_params::LearnParams;
use super::util::{Blob, WsBlob};

// Train/Test Impl
pub struct Solver {
    learn_rate: f32,
    alpha: f32,
    layers: LayersStorage,
}

impl Solver {
    pub fn new() -> Self {
        Self {
            layers: LayersStorage::new(),
            learn_rate: 0.4,
            alpha: 0.4,
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn setup_simple_network(&mut self, layers: &Vec<usize>) {
        if layers.len() < 3 {
            eprintln!("Invalid layers length !!!");
            return;
        }

        for (idx, val) in layers.iter().enumerate() {
            if idx == 0 {
                let l = Box::new(InputDataLayer::new(*val));
                self.layers.add_layer(l);
                continue;
            }
            if idx == layers.len() - 1 {
                let l = Box::new(ErrorLayer::new(*val, layers[idx - 1]));
                self.layers.add_layer(l);
                continue;
            }

            let l: Box<dyn AbstractLayer> = Box::new(HiddenLayer::new(*val, layers[idx - 1]));
            self.layers.add_layer(l);
        }
    }

    pub fn setup_network(&mut self, layers: LayersStorage) {
        self.layers = layers;
    }

    pub fn perform_step(&mut self, data: &DataBatch) {
        self.feedforward(&data, false);
        self.backpropagate(&data);
        self.optimize_network();
    }

    pub fn feedforward(&mut self, train_data: &DataBatch, print_out: bool) {
        let input_data = &train_data.input;

        let mut out = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            // handle input layer
            if idx == 0 {
                let result_out = l.forward(&input_data);

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

            let result_out = l.forward(out.unwrap());

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            };
        }

        let out_val = &out.unwrap()[0];

        if print_out {
            for i in out_val.iter() {
                println!("out val : {}", i);
            }
        }
    }

    pub fn backpropagate(&mut self, train_data: &DataBatch) {
        let expected_data = &train_data.expected;

        let mut out = None;

        for (idx, l) in self.layers.iter_mut().rev().enumerate() {
            if idx == 0 {
                let result_out = l.backward(expected_data, &WsBlob::new());

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

            let result_out = l.backward(out.unwrap().0, out.unwrap().1);

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

    pub fn optimize_network(&mut self) {
        let mut prev_lr = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            if idx == 0 {
                prev_lr = l.learn_params();
                continue;
            }

            debug!("current optimizing layer : {}", l.layer_type());

            let mut hidden_show = false;

            if l.layer_type() == "HiddenLayer" {
                hidden_show = true;
            }

            l.optimize(
                &|lr: &mut LearnParams, prev_lr: &LearnParams| {
                    let prev_vec = &prev_lr.output[0];

                    for neu_idx in 0..lr.output[0].len() {
                        for prev_idx in 0..prev_vec.len() {
                            let cur_ws_idx = [neu_idx, prev_idx];
                            // ALPHA
                            lr.ws[0][cur_ws_idx] += self.alpha * lr.ws_delta[0][cur_ws_idx];
                            // LEARNING RATE
                            lr.ws_delta[0][cur_ws_idx] =
                                self.learn_rate * lr.err_vals[0][neu_idx] * prev_vec[prev_idx];

                            lr.ws[0][cur_ws_idx] += lr.ws_delta[0][cur_ws_idx];
                        }
                    }
                },
                prev_lr.unwrap(),
            );

            prev_lr = l.learn_params();
        }
    }
}

impl Serialize for Solver {
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
