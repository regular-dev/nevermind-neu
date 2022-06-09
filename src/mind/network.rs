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
    test_val: f32,
}

impl Network {
    pub fn new(dataloader: Box<dyn DataLoader>) -> Self {
        Network {
            dataloader,
            layers: Vec::new(),
            test_val: 54.0,
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn setup_network(&mut self, layers: &Vec<usize>) {
        if layers.len() < 3 {
            eprintln!("Invalid layers length !!!");
            return;
        }

        // let mut cur_layer: &mut Box<dyn AbstractLayer> = &mut self.input_layer;

        for (idx, val) in layers.iter().enumerate() {
            if idx == 0 {
                let l: Box<dyn AbstractLayer> = Box::new(InputDataLayer::new(*val));
                self.layers.push(l);
                continue;
            }
            if idx == layers.len() - 1 {
                let mut l = Box::new(ErrorLayer::new(*val, layers[idx - 1]));
                self.layers.push(l);
                continue;
            }

            let mut l: Box<dyn AbstractLayer> = Box::new(HiddenLayer::new(*val, layers[idx - 1]));
            self.layers.push(l);
        }
    }

    fn get_train_step_data(&mut self) -> DataBatch {
        return self.dataloader.next().clone();
    }

    pub fn perform_step(&mut self) {
        let data = self.get_train_step_data();

        self.feedforward(&data);
        self.backpropagate(&data);
    }

    fn feedforward(&mut self, train_data: &DataBatch) {
        let input_data = train_data.input.clone();

        let mut out = None;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            // handle input layer
            if (idx == 0) {
                out = Some(l.forward(&input_data));
                continue;
            }

            out = Some(l.forward(out.unwrap()));
        }

        let out_val = &out.unwrap()[0];

        for i in out_val.iter() {
            println!("out val : {}", i);
        }
    }

    fn backpropagate(&mut self, train_data: &DataBatch) {
    }
}
