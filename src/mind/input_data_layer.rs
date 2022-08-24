use std::collections::HashMap;
use std::vec::Vec;

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerError, LayerForwardResult};

use super::activation::sigmoid_on_vec;
use super::learn_params::LearnParams;
use super::util::{Blob, DataVec, Variant, WsBlob, WsMat};

pub struct InputDataLayer {
    pub input_size: usize,
    pub lr: LearnParams,
}

impl AbstractLayer for InputDataLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult {
        let in_vec = &input[0];
        let out_vec = &mut self.lr.output;

        if in_vec.len() != self.input_size {
            eprintln!("Invalid input size for InputDataLayer : {}", in_vec.len());
            return Err(LayerError::InvalidSize);
        }

        sigmoid_on_vec(in_vec, out_vec);
        Ok(vec![&self.lr.output])
    }

    fn backward(
        &mut self,
        prev_input: Option<&Blob>,
        input: Option<&Blob>,
        weights: Option<&WsBlob>,
    ) -> LayerBackwardResult {
        Ok((&self.lr.output, &self.lr.ws))
    }

    fn layer_type(&self) -> &str {
        "InputDataLayer"
    }

    fn learn_params(&mut self) -> Option<&mut LearnParams> {
        Some(&mut self.lr)
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
            lr: LearnParams::new_only_output(input_size),
        }
    }
}
