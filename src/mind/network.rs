use std::vec::Vec;
use std::collections::HashMap;

use serde::{Serialize, Deserialize, Serializer};
use serde::ser::{SerializeStruct, SerializeSeq};
use serde_json;
use serde_yaml;

use log::{ info, error, debug, warn };

use std::fs::File;
use std::io::{Write, BufReader, BufRead, Error, ErrorKind};

use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::util::{Blob, Variant, DataVec};
use crate::mind::dataset::DataLoader;
use crate::mind::dummy_layer::DummyLayer;
use crate::mind::error_layer::ErrorLayer;
use crate::mind::hidden_layer::HiddenLayer;
use crate::mind::input_data_layer::InputDataLayer;

use super::{
    dataset::{DataBatch, SimpleDataLoader},
    input_data_layer,
};

/// Neural-Network
pub struct Network {
    dataloader: Box<dyn DataLoader>,
    solver: Solver,
}

struct Layers
{
    layers: Vec<Box<dyn AbstractLayer>>
}

// Train/Test Impl
struct Solver {
    lr: f32,
    alpha: f32,
    layers: Layers,
}

impl Network {
    pub fn new(dataloader: Box<dyn DataLoader>) -> Self {
        debug!("Created an neural network!");

        Network {
            dataloader,
            solver: Solver::new()
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    /// TODO : make this function static and make static constructor for Network class
    pub fn simple_setup_network(&mut self, layers: &Vec<usize>) {
        self.solver.setup_network(layers);
    }

    // pub fn load_network_cfg(&mut self, path: &str) -> Result<()> {


    //     Ok(())
    // }

    pub fn save_network_cfg(&mut self, path: &str) -> std::io::Result<()> {
        let json_str_result = serde_yaml::to_string(&self.solver);

        let mut output = File::create(path)?;

        match json_str_result {
            Ok(json_str) => {
                output.write_all(json_str.as_bytes())?;
            }, Err(x) => {
                eprintln!("Error (serde-yaml) serializing net layers !!!");
                return Err(std::io::Error::new(ErrorKind::Other, x));
            }
        }

        Ok(())
    }

    pub fn feedforward(&mut self, train_data: &DataBatch, print_out: bool) {
        self.solver.feedforward(train_data, print_out);
    }

    fn perform_step(&mut self) {
        let data = self.dataloader.next();
        self.solver.perform_step(&data);
    }

    pub fn train_for_n_times(&mut self, times: i64) {
        for _i in 0..times {
            self.perform_step();
        }
    }
}

impl Solver {
    fn new() -> Self {
        Self {
            layers: Layers::new(),
            lr: 0.1,
            alpha: 0.1,
        }
    }

     /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
     fn setup_network(&mut self, layers: &Vec<usize>) {
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

    fn perform_step(&mut self, data: &DataBatch) {
        self.feedforward(&data, false);
        self.backpropagate(&data);
        self.correct_weights();
    }

    fn feedforward(&mut self, train_data: &DataBatch, print_out: bool) {
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

    fn backpropagate(&mut self, train_data: &DataBatch) {
        let expected_data = &train_data.expected;

        let mut out = None;

        for (idx, l) in self.layers.iter_mut().rev().enumerate() {
            if idx == 0 {
                let result_out = l.backward(expected_data, &Blob::new());

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

    fn correct_weights(&mut self) {
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

impl Layers
{
    fn new() -> Self {
        Layers {
            layers: Vec::new(),
        }
    }

    fn iter_mut(&mut self) -> std::slice::IterMut< Box<dyn AbstractLayer> > {
        return self.layers.iter_mut();
    }

    fn add_layer(&mut self, l: Box<dyn AbstractLayer>) {
        self.layers.push(l);
    }
}

impl Serialize for Layers {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut layers_cfg = serializer.serialize_seq(Some(self.layers.len()))?;
        
        for l in self.layers.iter() {
            let l_cfg = l.layer_cfg();
            layers_cfg.serialize_element(&l_cfg)?;
        }

        layers_cfg.end()
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
