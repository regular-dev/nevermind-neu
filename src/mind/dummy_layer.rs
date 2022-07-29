use std::option::Option;

use crate::mind::abstract_layer::{AbstractLayer, LayerForwardResult, LayerBackwardResult};
use super::util::{Blob, Variant, DataVec, WsBlob, WsMat};

// not used
pub struct DummyLayer {
    output: Blob,
    fake_ws: WsBlob,
}

impl AbstractLayer for DummyLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult
    {
        Ok(&self.output)
    }
    fn backward(&mut self, input: &Blob, weights: &WsBlob) -> LayerBackwardResult
    {
        Ok((&self.output, &self.fake_ws))
    }

    fn optimize(&mut self, _prev_out: &Blob) -> &Blob {
        &self.output
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
            output: Blob::new(),
            fake_ws: vec![WsMat::zeros((0, 0))],
        }
    }
}