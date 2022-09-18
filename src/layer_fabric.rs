use std::collections::HashMap;

use log::warn;

use super::layers::{AbstractLayer, DummyLayer, ErrorLayer, HiddenLayer, InputDataLayer};
use super::util::Variant;
use crate::activation::Activation;
use crate::activation::*;


/// Fabric used to create neural network layers, when deserialing and other cases
/// TODO : create a macros for below implementation
pub fn create_layer(
    layer_type: &str,
    cfg: Option<&HashMap<String, Variant>>,
) -> Option<Box<dyn AbstractLayer>> {
    match layer_type {
        "ErrorLayer" => {
            let mut l = Box::new(ErrorLayer::default());
            if cfg.is_some() {
                l.set_layer_cfg(cfg.unwrap());
            }
            return Some(l);
        }
        "HiddenLayer" => {
            let mut l = Box::new(layers_macros::sigmoid_hidden_layer!());
            if cfg.is_some() {
                l.set_layer_cfg(cfg.unwrap());
            }
            return Some(l);
        }
        "InputDataLayer" => {
            let mut l = Box::new(InputDataLayer::default());
            if cfg.is_some() {
                l.set_layer_cfg(cfg.unwrap());
            }
            return Some(l);
        }
        "DummyLayer" => {
            let l = Box::new(DummyLayer::default());
            return Some(l);
        }
        _ => {
            warn!("Couldn't create a layer with name : {}", &layer_type);
            return None;
        }
    }
}

pub mod layers_macros {
    macro_rules! sigmoid_hidden_layer {
        (  ) => {
            {
                HiddenLayer::new(0, 0, activation_macros::sigmoid_activation!())                
            }
        };
    }

    macro_rules! tanh_hidden_layer {
        (  ) => {
            {
                HiddenLayer::new(0, 0, activation_macros::tanh_activation!())                
            }
        };
    }

    pub(crate) use sigmoid_hidden_layer; 
    pub(crate) use tanh_hidden_layer;
}
