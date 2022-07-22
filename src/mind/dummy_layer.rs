use std::option::Option;

use crate::mind::abstract_layer::{AbstractLayer, LayerForwardResult, LayerBackwardResult};
use super::util::{Blob, Variant, DataVec};

// not used
pub struct DummyLayer {
    output: Blob,
}

impl AbstractLayer for DummyLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult
    {
        Ok(&self.output)
    }
    fn backward(&mut self, input: &Blob, weights: &Blob) -> LayerBackwardResult
    {
        Ok((&self.output, &self.output))
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
            output: Blob::new()
        }
    }
}