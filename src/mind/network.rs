use std::error::Error;
use std::vec::Vec;

use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;
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
    layers: Vec<Box<dyn AbstractLayer>>,
}

impl Network {
    pub fn new(dataloader: Box<dyn DataLoader>) -> Self {
        Network {
            dataloader,
            layers: Vec::new(),
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn setup_network(&mut self, layers: &Vec<usize>) {
        if layers.len() < 3 {
            eprintln!("Invalid layers length !!!");
            return;
        }

        for (idx, val) in layers.iter().enumerate() {
            if idx == 0 {
                let l= Box::new(InputDataLayer::new(*val));
                self.layers.push(l);
                continue;
            }
            if idx == layers.len() - 1 {
                let l = Box::new(ErrorLayer::new(*val, layers[idx - 1]));
                self.layers.push(l);
                continue;
            }

            let l: Box<dyn AbstractLayer> = Box::new(HiddenLayer::new(*val, layers[idx - 1]));
            self.layers.push(l);
        }
    }

    fn get_train_step_data(&mut self) -> DataBatch {
        return self.dataloader.next().clone();
    }

    pub fn perform_step(&mut self) {
        let data = self.get_train_step_data();

        self.feedforward(&data, false);
        self.backpropagate(&data);
        self.feedforward(&data, false);
        self.correct_weights();
    }

    pub fn train_for_n_times(&mut self, times: i64) {
        for _i in 0..times {
            self.perform_step();
        }
    }

    pub fn feedforward(&mut self, train_data: &DataBatch, print_out: bool) {
        let input_data = &train_data.input;

        let mut out = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            // handle input layer
            if idx == 0 {
                out = Some(l.forward(&input_data));
                continue;
            }

            out = Some(l.forward(out.unwrap()));
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
                out = Some(l.backward(expected_data, &Blob::new()));
                continue;
            }

            out = Some(l.backward(out.unwrap().0, out.unwrap().1));
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
