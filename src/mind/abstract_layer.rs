use std::vec::Vec;
use std::collections::HashMap;

use serde::{Serialize, Deserialize, Serializer};
use serde::ser::{SerializeSeq, SerializeStruct};

use crate::mind::util::{Blob, DataVec, Variant};

pub enum LayerError {
    InvalidSize,
    OtherError
}

pub type LayerForwardResult<'a> = Result< &'a Blob, LayerError >;
pub type LayerBackwardResult<'a> = Result< (&'a Blob, &'a Blob), LayerError >;

pub trait AbstractLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult;
    /// returns out_values and array of weights
    fn backward(&mut self, input: &Blob, weights: &Blob) -> LayerBackwardResult;
    fn optimize(&mut self, prev_out: &Blob) -> &Blob;

    fn layer_type(&self) -> &str;

    fn size(&self) -> usize;

    fn layer_cfg(&self) -> HashMap< &str, Variant > { 
        let mut cfg: HashMap<&str, Variant> = HashMap::new();

        cfg.insert("layer_type", Variant::String(String::from(self.layer_type())));

        cfg
    }

    fn set_layers_cfg(&self, _cfg: HashMap< String, Variant >) {  }
}