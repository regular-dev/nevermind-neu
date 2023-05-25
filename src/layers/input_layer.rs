use std::collections::HashMap;

use super::abstract_layer::{AbstractLayer, LayerError, LayerForwardResult};

use log::error;

use crate::learn_params::LearnParams;
use crate::util::{Array2D, Blob, DataVecPtr, Variant, WithParams};

#[derive(Default, Clone)]
pub struct InputLayer {
    pub input_size: usize,
    pub lr_params: LearnParams,
}

impl AbstractLayer for InputLayer {
    fn forward_input(&mut self, input: Array2D) -> LayerForwardResult {
        if input.ncols() != self.input_size {
            error!(
                "Invalid input size for InputDataLayer : {}",
                input.shape()[1]
            );
            return Err(LayerError::InvalidSize);
        }
        
        *self.lr_params.output.borrow_mut() = input;

        Ok(vec![self.lr_params.clone()])
    }

    fn layer_type(&self) -> &str {
        "InputLayer"
    }

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.lr_params.clone())
    }

    fn set_learn_params(&mut self, lp: LearnParams) {
        self.lr_params = lp;
    }

    fn set_input_shape(&mut self, sh: &[usize]) { }

    fn size(&self) -> usize {
        self.input_size
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = InputLayer::new(self.input_size);
        copy_l.set_learn_params(self.lr_params.copy());
        Box::new(copy_l)
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }
}

impl InputLayer {
    pub fn new(input_size: usize) -> Self {
        Self {
            input_size,
            lr_params: LearnParams::empty(),
        }
    }

    pub fn new_box(size: usize) -> Box<Self> {
        Box::new(InputLayer::new(size))
    }
}

impl WithParams for InputLayer {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg: HashMap<String, Variant> = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.input_size as i32));

        cfg
    }

    fn set_cfg(&mut self, cfg: &HashMap<String, Variant>) {
        let mut size: usize = 0;

        if let Variant::Int(var_size) = cfg.get("size").unwrap() {
            size = *var_size as usize;
        }

        if size > 0 {
            self.input_size = size;
            self.lr_params = LearnParams::empty();
        }
    }
}
