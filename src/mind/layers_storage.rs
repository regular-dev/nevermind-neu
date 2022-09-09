use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

use super::abstract_layer::AbstractLayer;
use super::input_data_layer::InputDataLayer;
use super::error_layer::ErrorLayer;
use super::hidden_layer::HiddenLayer;

pub struct LayersStorage {
    layers: Vec<Box<dyn AbstractLayer>>,
}

impl LayersStorage {
    pub fn new() -> Self {
        LayersStorage { layers: Vec::new() }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn new_simple_network(layers: &Vec<usize>) -> Self {
        let mut ls = LayersStorage::new();

        if layers.len() < 3 {
            eprintln!("Invalid layers length !!!");
            return LayersStorage::new();
        }

        for (idx, val) in layers.iter().enumerate() {
            if idx == 0 {
                let l = Box::new(InputDataLayer::new(*val));
                ls.add_layer(l);
                continue;
            }
            if idx == layers.len() - 1 {
                let l = Box::new(ErrorLayer::new(*val, layers[idx - 1]));
                ls.add_layer(l);
                continue;
            }

            let l: Box<dyn AbstractLayer> = Box::new(HiddenLayer::new(*val, layers[idx - 1]));
            ls.add_layer(l);
        }

        ls
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Box<dyn AbstractLayer>> {
        return self.layers.iter_mut();
    }

    pub fn add_layer(&mut self, l: Box<dyn AbstractLayer>) {
        self.layers.push(l);
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn at_mut(&mut self, id: usize) -> &mut Box<dyn AbstractLayer> {
        &mut self.layers[id]
    }

    pub fn at(&self, id: usize) -> &Box<dyn AbstractLayer> {
        &self.layers[id]
    }
}

impl Serialize for LayersStorage {
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
