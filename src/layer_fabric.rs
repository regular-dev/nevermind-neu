use std::collections::HashMap;

use log::warn;

use crate::util::*;
use crate::layers::*;
use crate::util::*;

/// Fabric used to create neural network layers, when deserialing and other cases
pub fn create_layer(
    layer_type: &str,
    cfg: Option<&HashMap<String, Variant>>,
) -> Option<Box<dyn AbstractLayer>> {
    match layer_type {
        "EuclideanLossLayer" => {
            if let Some(cfg_val) = cfg {
                let mut l: Box<dyn AbstractLayer>;
                let activation = cfg_val.get("activation").unwrap();

                if let Variant::String(activation) = activation {
                    if activation == "sigmoid" {
                        l = Box::new(layers_macros::sigmoid_error_layer!());
                    } else if activation == "tanh" {
                        l = Box::new(layers_macros::tanh_error_layer!());
                    } else {
                        l = Box::new(layers_macros::raw_error_layer!());
                    }
                } else {
                    l = Box::new(layers_macros::raw_error_layer!());
                }

                l.set_cfg(cfg_val);

                return Some(l);
            } else {
                let l = Box::new(layers_macros::raw_error_layer!());
                return Some(l);
            }
        }
        "FcLayer" => {
            if let Some(cfg_val) = cfg {
                let mut l: Box<dyn AbstractLayer>;
                let activation = cfg_val.get("activation").unwrap(); // TODO : handle unwrap()

                if let Variant::String(activation) = activation {
                    if activation == "sigmoid" {
                        l = Box::new(layers_macros::sigmoid_fc_layer!());
                    } else if activation == "tanh" {
                        l = Box::new(layers_macros::tanh_fc_layer!());
                    } else if activation == "relu" {
                        l = Box::new(layers_macros::relu_fc_layer!());
                    } else if activation == "leaky_relu" {
                        l = Box::new(layers_macros::leaky_relu_fc_layer!());
                    } else {
                        l = Box::new(layers_macros::raw_rc_layer!());
                    }
                } else {
                    l = Box::new(layers_macros::raw_rc_layer!());
                }

                l.set_cfg(cfg_val);
                return Some(l);
            }

            let l = Box::new(layers_macros::sigmoid_fc_layer!());
            return Some(l);
        }
        "InputLayer" => {
            let mut l = Box::new(InputLayer::default());
            if cfg.is_some() {
                l.set_cfg(cfg.unwrap());
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
                l.set_cfg(cfg_val);
            }
            return Some(l);
        }
        _ => {
            warn!("Couldn't create a layer with name : {}", &layer_type);
            return None;
        }
    }
}

#[cfg(feature = "opencl")]
pub fn create_layer_ocl(
    layer_type: &str,
    cfg: Option<&HashMap<String, Variant>>,
) -> Option<Box<dyn AbstractLayerOcl>> {
    match layer_type {
        "InputLayerOcl" => {
            let mut l = Box::new(InputLayerOcl::default());
            if let Some(cfg_val) = cfg {
                l.set_cfg(cfg_val);
            }
            return Some(l);
        },
        "FcLayerOcl" => {
            let mut l = Box::new(FcLayerOcl::default());
            if let Some(cfg_val) = cfg {
                l.set_cfg(cfg_val);
            }
            return Some(l);
        },
        "EuclideanLossLayerOcl" => {
            let mut l = Box::new(EuclideanLossLayerOcl::default());
            if let Some(cfg_val) = cfg {
                l.set_cfg(cfg_val);
            }
            return Some(l);
        }

        _ => {return None;}
    }
}

pub mod layers_macros {
    macro_rules! sigmoid_fc_layer {
        (  ) => {{
            FcLayer::new(0, activation_macros::sigmoid_activation!())
        }};
    }

    macro_rules! tanh_fc_layer {
        (  ) => {{
            FcLayer::new(0, activation_macros::tanh_activation!())
        }};
    }

    macro_rules! relu_fc_layer {
        () => {
            FcLayer::new(0, activation_macros::relu_activation!())
        };
    }

    macro_rules! leaky_relu_fc_layer {
        () => {
            FcLayer::new(0, activation_macros::leaky_relu_activation!())
        };
    }

    macro_rules! raw_rc_layer {
        () => {
            FcLayer::new(0, activation_macros::raw_activation!())
        };
    }

    macro_rules! sigmoid_error_layer {
        () => {
            EuclideanLossLayer::new(0, activation_macros::sigmoid_activation!())
        };
    }

    macro_rules! tanh_error_layer {
        () => {
            EuclideanLossLayer::new(0, activation_macros::tanh_activation!())
        };
    }

    macro_rules! raw_error_layer {
        () => {
            EuclideanLossLayer::new(0, activation_macros::raw_activation!())
        };
    }

    pub(crate) use leaky_relu_fc_layer;
    pub(crate) use raw_error_layer;
    pub(crate) use raw_rc_layer;
    pub(crate) use relu_fc_layer;
    pub(crate) use sigmoid_error_layer;
    pub(crate) use sigmoid_fc_layer;
    pub(crate) use tanh_error_layer;
    pub(crate) use tanh_fc_layer;
}
