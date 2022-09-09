use std::collections::HashMap;

use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer, Deserialize, Deserializer};

use super::abstract_layer::AbstractLayer;
use super::input_data_layer::InputDataLayer;
use super::error_layer::ErrorLayer;
use super::hidden_layer::HiddenLayer;
use super::util::Variant;

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

/// Helper class to easy ser/deserialize
#[derive(Serialize, Deserialize)]
struct SerdeLayerParam {
    name: String,
    params: HashMap<String, Variant>
}

/// Helper class to easy ser/deserialize
#[derive(Serialize, Deserialize, Default)]
struct SerdeLayersStorage {
    layers_cfg: Vec< SerdeLayerParam >,
}

impl Serialize for LayersStorage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s_layers_storage = SerdeLayersStorage::default();

        for l in self.layers.iter() {
            let s_layer_param = SerdeLayerParam {
                name: l.layer_type().to_owned(),
                params: l.layer_cfg(),
            };
            s_layers_storage.layers_cfg.push(s_layer_param);
        }

        s_layers_storage.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LayersStorage {
    fn deserialize<D>(deserializer: D) -> Result<LayersStorage, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s_layers_storage = SerdeLayersStorage::deserialize(deserializer)?;

        // TODO : impl with layer fabric
        let todo_ls = LayersStorage { layers: Vec::new() };

        Ok(todo_ls)
    }
}

