use std::collections::HashMap;
use std::vec::Vec;

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use super::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerError, LayerForwardResult};

use super::activation::sigmoid_on_vec;
use super::learn_params::{LearnParams, ParamsBlob};
use super::util::{Blob, DataVec, Variant, WsBlob, WsMat};

#[derive(Default)]
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

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.lr_params.clone())
    }

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let mut cfg: HashMap<String, Variant> = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.input_size as i32));

        cfg
    }

    fn set_layer_cfg(&mut self, cfg: &HashMap<String, Variant>) {
        let mut size : usize = 0;

        if let Variant::Int(var_size) = cfg.get("size").unwrap() {
            size = *var_size as usize;
        }

        if size > 0 {
            self.input_size = size;
            self.lr_params = LearnParams::new_only_output(size);
        }
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
