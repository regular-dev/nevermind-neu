use std::collections::HashMap;

use crate::dataloader::DataBatch;
use crate::learn_params::{LearnParams, ParamsBlob};
use crate::util::{Batch, DataVec, DataVecPtr, Variant};

#[derive(Debug)]
pub enum LayerError {
    InvalidSize,
    OtherError,
    NotImpl,
}

pub type LayerForwardResult = Result<ParamsBlob, LayerError>;
pub type LayerBackwardResult = Result<ParamsBlob, LayerError>;

pub trait AbstractLayer {
    // for signature for input layers
    fn forward_input(&mut self, input_data: Batch) -> LayerForwardResult {
        Err(LayerError::NotImpl)
    }

    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        Err(LayerError::NotImpl)
    }

    /// returns out_values and array of weights
    fn backward(&mut self, prev_input: ParamsBlob, input: ParamsBlob) -> LayerBackwardResult {
        Err(LayerError::NotImpl)
    }

    fn backward_output(&mut self, prev_input: ParamsBlob, expected: Batch) -> LayerBackwardResult {
        Err(LayerError::NotImpl)
    }

    fn layer_type(&self) -> &str;

    fn size(&self) -> usize;

    fn learn_params(&self) -> Option<LearnParams>;
    fn set_learn_params(&mut self, lp: LearnParams);

    fn set_input_shape(&mut self, sh: &[usize]);

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let cfg: HashMap<String, Variant> = HashMap::new();
        cfg
    }

    fn set_layer_cfg(&mut self, _cfg: &HashMap<String, Variant>) {}

    // Do copy layer memory(ws, output, ...)
    fn copy_layer(&self) -> Box<dyn AbstractLayer>;

    // Do copy only Rc
    fn clone_layer(&self) -> Box<dyn AbstractLayer>;
}
