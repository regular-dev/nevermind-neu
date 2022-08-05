use std::option::Option;

use crate::mind::abstract_layer::{AbstractLayer, LayerForwardResult, LayerBackwardResult};
use super::util::{Blob, Variant, DataVec, WsBlob, WsMat};
use super::learn_params::LearnParams;

// not used
pub struct DummyLayer {
    output: DataVec,
    fake_lr: LearnParams,
}

impl AbstractLayer for DummyLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult
    {
        Ok(&self.output)
    }
    fn backward(&mut self, input: &Blob, weights: &WsBlob) -> LayerBackwardResult
    {
        Ok((&self.output, &self.fake_lr.ws))
    }

    fn learn_params(&mut self) -> Option< &mut LearnParams > {
        Some(&mut self.fake_lr)
    }

    fn layer_type(&self) -> &str {
        return "DummyLayer";
    }

    fn size(&self) -> usize 
    {
        0
    }
}

impl DummyLayer {
    pub fn new() -> Self {
        DummyLayer{
            output: DataVec::zeros(0),
            fake_lr: LearnParams::new(0, 0),
        }
    }
}