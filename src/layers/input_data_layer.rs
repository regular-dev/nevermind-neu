use std::collections::HashMap;

use super::abstract_layer::{AbstractLayer, LayerError, LayerForwardResult};

use crate::learn_params::LearnParams;
use crate::util::{Batch, Blob, DataVecPtr, Variant};

#[derive(Default, Clone)]
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

    fn set_learn_params(&mut self, lp: LearnParams) {
        self.lr_params = lp;
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
            self.lr_params = LearnParams::empty();
        }
    }

    fn set_input_shape(&mut self, sh: &[usize]) { }

    fn size(&self) -> usize {
        self.input_size
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = InputDataLayer::new(self.input_size);
        copy_l.set_learn_params(self.lr_params.copy());
        Box::new(copy_l)
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }
}

impl InputDataLayer {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            lr_params: LearnParams::empty(),
        }
    }
}
