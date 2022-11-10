use std::collections::HashMap;

use std::fmt;

use log::debug;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::layer_fabric::*;
use super::layers::AbstractLayer;
use super::layers::ErrorLayer;
use super::layers::HiddenLayer;
use super::layers::InputDataLayer;
use super::util::Variant;

use crate::activation::*;

#[derive(Default)]
pub struct LayersStorage {
    layers: Vec<Box<dyn AbstractLayer>>,
}

impl LayersStorage {
    pub fn empty() -> Self {
        LayersStorage { layers: Vec::new() }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn new_simple_network(layers: &Vec<usize>) -> Self {
        let mut ls = LayersStorage::empty();

        if layers.len() < 3 {
            eprintln!("Invalid layers length !!!");
            return LayersStorage::empty();
        }

        for (idx, val) in layers.iter().enumerate() {
            if idx == 0 {
                let l = Box::new(InputDataLayer::new(*val));
                ls.add_layer(l);
                continue;
            }
            if idx == layers.len() - 1 {
                let l = Box::new(ErrorLayer::new(
                    *val,
                    layers[idx - 1],
                    activation_macros::raw_activation!(),
                ));
                ls.add_layer(l);
                continue;
            }

            let l: Box<dyn AbstractLayer> = Box::new(HiddenLayer::new(
                *val,
                layers[idx - 1],
                activation_macros::tanh_activation!(),
            ));
            ls.add_layer(l);
        }

        ls
    }

    pub fn fit_to_batch_size(&mut self, batch_size: usize) {
        for i in &self.layers {
            let mut lr = i.learn_params().unwrap();
            lr.fit_to_batch_size(batch_size);
        }
    }

    pub fn prepare_for_tests(&self, batch_size: usize) {
        for i in &self.layers {
            let mut lr = i.learn_params().unwrap();
            lr.prepare_for_tests(batch_size);
        }
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

    pub fn first(&self) -> Option<&Box<dyn AbstractLayer>> {
        self.layers.first()
    }

    pub fn first_mut(&mut self) -> Option<&mut Box<dyn AbstractLayer>> {
        self.layers.first_mut()
    }

    pub fn last(&self) -> Option<&Box<dyn AbstractLayer>> {
        self.layers.last()
    }

    pub fn last_mut(&mut self) -> Option<&mut Box<dyn AbstractLayer>> {
        self.layers.last_mut()
    }
}

impl fmt::Display for LayersStorage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut out = String::new();

        for i in &self.layers {
            out += i.size().to_string().as_str();
            out += "-";
        }

        write!(f, "{}", &out.as_str()[0..out.len() - 1])
    }
}

/// Helper class to easy ser/deserialize
#[derive(Serialize, Deserialize)]
pub struct SerdeLayerParam {
    name: String,
    params: HashMap<String, Variant>,
}

/// Helper class to easy ser/deserialize
#[derive(Serialize, Deserialize, Default)]
pub struct SerdeLayersStorage {
    layers_cfg: Vec<SerdeLayerParam>,
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

        let mut ls = LayersStorage { layers: Vec::new() };

        for i in &s_layers_storage.layers_cfg {
            let l_opt = create_layer(i.name.as_str(), Some(&i.params));

            if let Some(l) = l_opt {
                debug!("Create layer : {}", i.name);
                ls.layers.push(l);
            } else {
                // TODO : impl return D::Error
                panic!("Bad deserialization");
            }
        }

        Ok(ls)
    }
}

impl Clone for LayersStorage {
    fn clone(&self) -> Self {
        let mut ls = LayersStorage::empty();

        for i in &self.layers {
            ls.add_layer(i.clone_layer());
        }

        ls
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}
