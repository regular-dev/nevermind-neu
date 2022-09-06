use std::collections::HashMap;
use std::vec::Vec;

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerError, LayerForwardResult};

use super::activation::sigmoid_on_vec;
use super::learn_params::{LearnParams, ParamsBlob};
use super::util::{Blob, DataVec, Variant, WsBlob, WsMat};

pub struct InputDataLayer {
    pub input_size: usize,
    pub lr_params: LearnParams,
}

impl AbstractLayer for InputDataLayer {
    fn forward_input(&mut self, input: &DataVec) -> LayerForwardResult {
        let out_vec = &mut self.lr_params.output.borrow_mut();

        if input.len() != self.input_size {
            eprintln!("Invalid input size for InputDataLayer : {}", input.len());
            return Err(LayerError::InvalidSize);
        }

        sigmoid_on_vec(input, out_vec);
        Ok(vec![self.lr_params.clone()])
    }

    fn layer_type(&self) -> &str {
        "InputDataLayer"
    }

    fn learn_params(&mut self) -> Option<LearnParams> {
        Some(self.lr_params.clone())
    }

    fn layer_cfg(&self) -> HashMap<&str, Variant> {
        let mut cfg: HashMap<&str, Variant> = HashMap::new();

        cfg.insert(
            "layer_type",
            Variant::String(String::from(self.layer_type())),
        );
        cfg.insert("size", Variant::Int(self.input_size as i32));

        cfg
    }

    fn size(&self) -> usize {
        self.input_size
    }
}

impl InputDataLayer {
    pub fn load_data(&mut self, input: Blob) {}

    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            lr_params: LearnParams::new_only_output(input_size),
        }
    }
}
