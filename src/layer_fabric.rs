use std::collections::HashMap;

use log::warn;

use super::util::Variant;
use crate::activation::Activation;
use crate::activation::*;
use crate::layers::*;

/// Fabric used to create neural network layers, when deserialing and other cases
pub fn create_layer(
    layer_type: &str,
    cfg: Option<&HashMap<String, Variant>>,
) -> Option<Box<dyn AbstractLayer>> {
    match layer_type {
        "ErrorLayer" => {
            let mut l = Box::new(layers_macros::raw_error_layer!());
            if cfg.is_some() {
                l.set_layer_cfg(cfg.unwrap());
            }
            return Some(l);
        }
        "HiddenLayer" => {
            // TODO : deserialize activation function
            if let Some(cfg_val) = cfg {
                let mut l: Option<Box<dyn AbstractLayer>> = None;
                let activation = cfg_val.get("activation").unwrap(); // TODO : refactor unwrap()

                if let Variant::String(activation) = activation {
                    if activation == "sigmoid" {
                        l = Some(Box::new(layers_macros::sigmoid_hidden_layer!()));
                    } else if activation == "tanh" {
                        l = Some(Box::new(layers_macros::tanh_hidden_layer!()));
                    } else if activation == "relu" {
                        l = Some(Box::new(layers_macros::relu_hidden_layer!()));
                    }
                }

                if l.is_none() {
                    l = Some(Box::new(layers_macros::sigmoid_hidden_layer!()));
                }

                l.as_mut().unwrap().set_layer_cfg(cfg_val);
                return l;
            }

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
        "SoftmaxLossLayer" => {
            let mut l = Box::new(SoftmaxLossLayer::default());
            if let Some(cfg_val) = cfg {
                l.set_layer_cfg(cfg_val);
            }
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
        (  ) => {{
            HiddenLayer::new(0, 0, activation_macros::sigmoid_activation!())
        }};
    }

    macro_rules! tanh_hidden_layer {
        (  ) => {{
            HiddenLayer::new(0, 0, activation_macros::tanh_activation!())
        }};
    }

    macro_rules! relu_hidden_layer {
        () => {
            HiddenLayer::new(0, 0, activation_macros::relu_activation!())
        };
    }

    macro_rules! sigmoid_error_layer {
        () => {
            ErrorLayer::new(0, 0, activation_macros::sigmoid_activation!())
        };
    }

    macro_rules! tanh_error_layer {
        () => {
            ErrorLayer::new(0, 0, activation_macros::tanh_activation!())
        };
    }

    macro_rules! raw_error_layer {
        () => {
            ErrorLayer::new(0, 0, activation_macros::raw_activation!())
        };
    }

    pub(crate) use raw_error_layer;
    pub(crate) use relu_hidden_layer;
    pub(crate) use sigmoid_error_layer;
    pub(crate) use sigmoid_hidden_layer;
    pub(crate) use tanh_error_layer;
    pub(crate) use tanh_hidden_layer;
}
