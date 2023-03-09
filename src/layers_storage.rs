use std::collections::HashMap;

use std::fmt;

use log::{debug, error};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use std::slice::IterMut;

use super::layer_fabric::*;
use super::layers::AbstractLayer;
use super::layers::EuclideanLossLayer;
use super::layers::FcLayer;
use super::layers::InputDataLayer;
use super::util::Variant;

use crate::activation::*;


pub trait LayersStorage {
    fn fit_to_batch_size(&mut self, batch_size: usize);
    fn prepare_for_tests(&mut self, batch_size: usize);
    fn iter_mut(&mut self) -> IterMut<Box<dyn AbstractLayer>>;
    
}

#[derive(Default)]
pub struct SequentialLayersStorage {
    layers: Vec<Box<dyn AbstractLayer>>,
}

impl SequentialLayersStorage {
    pub fn empty() -> Self {
        SequentialLayersStorage { layers: Vec::new() }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn new_simple_network(layers: &Vec<usize>) -> Self {
        let mut ls = SequentialLayersStorage::empty();

        if layers.len() < 3 {
            error!("Invalid layers length !");
            return SequentialLayersStorage::empty();
        }

        for (idx, val) in layers.iter().enumerate() {
            if idx == 0 {
                let l = Box::new(InputDataLayer::new(*val));
                ls.add_layer(l);
                continue;
            }
            if idx == layers.len() - 1 {
                let l = Box::new(EuclideanLossLayer::new(
                    *val,
                    activation_macros::raw_activation!(),
                ));
                ls.add_layer(l);
                continue;
            }

            let l: Box<dyn AbstractLayer> = Box::new(FcLayer::new(
                *val,
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

    pub fn prepare_for_tests(&mut self, batch_size: usize) {
        for i in self.layers.iter_mut() {
            let mut lr = i.learn_params().unwrap();
            lr.prepare_for_tests(batch_size);
            i.set_learn_params(lr);
        }
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Box<dyn AbstractLayer>> {
        return self.layers.iter_mut();
    }

    pub fn add_layer(&mut self, l: Box<dyn AbstractLayer>) {
        // if !self.layers.is_empty() {
        //     if let Variant::Int(l_prev_size) = l.layer_cfg()["prev_size"] {
        //         let last_prev_size = self.last().unwrap().size();
        //         if l_prev_size != last_prev_size as i32 {
        //             warn!("Previous size {} of new layer doesn't match {}", l_prev_size, last_prev_size);
        //         }
        //     }
        // }

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

impl fmt::Display for SequentialLayersStorage {
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
    pub name: String,
    pub params: HashMap<String, Variant>,
}

/// Helper class to easy ser/deserialize
#[derive(Serialize, Deserialize, Default)]
pub struct SerdeLayersStorage {
    pub layers_cfg: Vec<SerdeLayerParam>,
}

impl Serialize for SequentialLayersStorage {
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

impl<'de> Deserialize<'de> for SequentialLayersStorage {
    fn deserialize<D>(deserializer: D) -> Result<SequentialLayersStorage, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s_layers_storage = SerdeLayersStorage::deserialize(deserializer)?;

        let mut ls = SequentialLayersStorage::empty();

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

impl Clone for SequentialLayersStorage {
    fn clone(&self) -> Self {
        let mut ls = SequentialLayersStorage::empty();

        for i in &self.layers {
            ls.add_layer(i.clone_layer());
        }

        ls
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}
