use std::collections::HashMap;
use std::vec::Vec;

use ndarray::Array2;

use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};

use std::cell::RefCell;

use super::learn_params::{LearnParams, LearnParamsPtr, ParamsBlob};
use super::util::{Blob, DataVec, Num, Variant, WsBlob};

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
    fn forward_input(&mut self, input_data: &DataVec) -> LayerForwardResult {
        Err(LayerError::NotImpl)
    }

    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        Err(LayerError::NotImpl)
    }

    /// returns out_values and array of weights
    fn backward(&mut self, prev_input: ParamsBlob, input: ParamsBlob) -> LayerBackwardResult {
        Err(LayerError::NotImpl)
    }

    fn backward_output(
        &mut self,
        prev_input: ParamsBlob,
        expected: &DataVec,
    ) -> LayerBackwardResult {
        Err(LayerError::NotImpl)
    }

    fn layer_type(&self) -> &str;

    fn size(&self) -> usize;

    fn learn_params(&self) -> Option<LearnParams>;

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let mut cfg: HashMap<String, Variant> = HashMap::new();
        cfg
    }

    fn set_layer_cfg(&mut self, _cfg: &HashMap<String, Variant>) {}
}
