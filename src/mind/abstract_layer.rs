use std::collections::HashMap;
use std::vec::Vec;

use ndarray::Array2;

use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};

use super::learn_params::LearnParams;
use super::util::{Blob, DataVec, Num, Variant, WsBlob};

pub enum LayerError {
    InvalidSize,
    OtherError,
}

pub type LayerForwardResult<'a> = Result<&'a Blob, LayerError>;
pub type LayerBackwardResult<'a> = Result<(&'a Blob, &'a WsBlob), LayerError>;

pub trait AbstractLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult;
    /// returns out_values and array of weights
    fn backward(&mut self, input: &Blob, weights: &WsBlob) -> LayerBackwardResult;
    //fn optimize(&mut self, prev_out: &Blob) -> &Blob;

    fn layer_type(&self) -> &str;

    fn size(&self) -> usize;

    fn learn_params(&mut self) -> Option<&mut LearnParams>;

    fn layer_cfg(&self) -> HashMap<&str, Variant> {
        let mut cfg: HashMap<&str, Variant> = HashMap::new();

        cfg.insert(
            "layer_type",
            Variant::String(String::from(self.layer_type())),
        );

        cfg
    }

    fn optimize(&mut self, f: &dyn Fn(&mut LearnParams, &LearnParams), prev_lr: &LearnParams) {
        f(self.learn_params().unwrap(), prev_lr);
    }

    fn set_layers_cfg(&self, _cfg: HashMap<String, Variant>) {}
}
