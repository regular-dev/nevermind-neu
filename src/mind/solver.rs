use serde::{Serializer, Serialize};
use serde::ser::{SerializeStruct};

use super::util::{WsBlob, Blob};
use super::dataset::{DataBatch};
use super::abstract_layer::AbstractLayer;
use super::error_layer::ErrorLayer;
use super::input_data_layer::InputDataLayer;
use super::hidden_layer::HiddenLayer;
use super::layers_storage::LayersStorage;


// Train/Test Impl
pub struct Solver {
    lr: f32,
    alpha: f32,
    layers: LayersStorage,
}

impl Solver {
    pub fn new() -> Self {
        Self {
            layers: LayersStorage::new(),
            lr: 0.1,
            alpha: 0.1,
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
                    },
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
                },
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
                    },
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
                },
                Ok(val) => {
                    out = Some(val);
                }
            }
        }
    }

    pub fn optimize_network(&mut self) {
        let mut out = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            if idx == 0 {
                out = Some(l.optimize(&Blob::new()));
                continue;
            }
            out = Some(l.optimize(&out.unwrap()));
        }
    }
}

impl Serialize for Solver {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut solver_cfg = serializer.serialize_struct("Solver Configuration", 3)?;
        solver_cfg.serialize_field("learning_rate", &self.lr)?;
        solver_cfg.serialize_field("alpha", &self.alpha)?;
        solver_cfg.serialize_field("layers_cfg", &self.layers)?;
        solver_cfg.end()
    }
}