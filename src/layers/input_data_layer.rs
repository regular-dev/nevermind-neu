use std::collections::HashMap;

use super::abstract_layer::{AbstractLayer, LayerError, LayerForwardResult};

use crate::dataloader::DataBatch;
use crate::learn_params::LearnParams;
use crate::util::{Batch, Blob, DataVecPtr, Variant};

#[derive(Default)]
pub struct InputDataLayer {
    pub input_size: usize,
    pub lr_params: LearnParams,
}

impl AbstractLayer for InputDataLayer {
    fn forward_input(&mut self, input: Batch) -> LayerForwardResult {
        if input.ncols() != self.input_size {
            eprintln!(
                "Invalid input size for InputDataLayer : {}",
                input.shape()[1]
            );
            return Err(LayerError::InvalidSize);
        }

        *self.lr_params.output.borrow_mut() = input;

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
        let mut size: usize = 0;

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
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            lr_params: LearnParams::new_only_output(input_size),
        }
    }
}
