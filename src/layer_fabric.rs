use std::collections::HashMap;

use log::warn;

use super::layers::{AbstractLayer, DummyLayer, ErrorLayer, HiddenLayer, InputDataLayer};
use super::util::Variant;

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
            let mut l = Box::new(HiddenLayer::default());
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
