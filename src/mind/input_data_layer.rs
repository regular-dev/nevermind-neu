use std::vec::Vec;
use std::collections::HashMap;

use serde::{Serialize, Serializer};
use serde::ser::{SerializeStruct};

use super::abstract_layer::{
    AbstractLayer, LayerBackwardResult, LayerError, LayerForwardResult,
};

use super::util::{Blob, Variant, DataVec, WsBlob, WsMat};
use super::activation::sigmoid_on_vec;


pub struct InputDataLayer {
    pub input_size: usize,
    pub output: Blob,
    pub fake_ws: WsBlob,
}

impl AbstractLayer for InputDataLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult {
        let in_vec = &input[0];
        let out_vec = &mut self.output[0];

        if in_vec.len() != self.input_size {
            eprintln!("Invalid input size for InputDataLayer : {}", in_vec.len());
            return Err(LayerError::InvalidSize);
        }

        sigmoid_on_vec(in_vec, out_vec);
        Ok(&self.output)
    }
    fn backward(&mut self, input: &Blob, weights: &WsBlob) -> LayerBackwardResult {
        Ok((&self.output, &self.fake_ws))
    }

    fn optimize(&mut self, _prev_out: &Blob) -> &Blob {
        &self.output
    }

    fn layer_type(&self) -> &str {
        "InputDataLayer"
    }

    fn layer_cfg(&self) -> HashMap< &str, Variant > { 
        let mut cfg: HashMap<&str, Variant> = HashMap::new();

        cfg.insert("layer_type", Variant::String(String::from(self.layer_type())));
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
        let mut vec_outp = DataVec::zeros(input_size);

        Self {
            input_size,
            output: vec![vec_outp],
            fake_ws: vec![WsMat::zeros((0, 0))],
        }
    }
}
