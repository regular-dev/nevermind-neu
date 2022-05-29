use std::vec::Vec;

use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::dummy_layer::DummyLayer;
use crate::mind::input_data_layer::InputDataLayer;
use crate::mind::error_layer::ErrorLayer;
use crate::mind::hidden_layer::HiddenLayer;
use crate::mind::dataset::DataLoader;

use super::{dataset::{SimpleDataLoader, DataBatch}, input_data_layer};

/// Neural-Network
pub struct Network {
    input_layer: Box<dyn AbstractLayer>,
    output_layer: Box<dyn AbstractLayer>,
    dataloader: Box<dyn DataLoader>,
}

impl Network {
    pub fn new(dataloader: Box<dyn DataLoader>) -> Self 
    {   
        Network {
            input_layer: Box::new(DummyLayer::new()),
            output_layer: Box::new(DummyLayer::new()),
            dataloader
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn setupNetwork(&mut self, layers: &Vec<i32>)
    {
        if layers.len() < 3 {
            eprintln!("Invalid layers length !!!");
            return;
        }

        let mut cur_layer: &mut Box<dyn AbstractLayer> = &mut self.input_layer;

        for (idx,val) in layers.iter().enumerate() {
            if idx == 0 {
                let l : Box<dyn AbstractLayer> = Box::new(InputDataLayer::new(*val));
                self.input_layer = l;
                cur_layer = &mut self.input_layer;
            }
            if idx == layers.len() - 1 {
                let mut l = Box::new(ErrorLayer::new(*val));
                self.output_layer = l;
            }

            let mut l : Box<dyn AbstractLayer> = Box::new(HiddenLayer::new(*val));
            cur_layer.add_next_layer(l);
            
            cur_layer = cur_layer.next_layer(0).unwrap();
        }
    }

    fn perform_step(&mut self)
    {
        let data = self.dataloader.next();
        let input_data = data.input.clone();
        let expected_data = data.expected.clone();

        let mut cur_layer = self.input_layer.next_layer(0).unwrap();
        let mut out = cur_layer.forward(input_data);

        loop {
            if cur_layer.next_layer(0).is_none() {
                break;
            }

            let next_layer = cur_layer.next_layer(0).unwrap();

            out = next_layer.forward(out);
            cur_layer = next_layer;
        }

    }


}