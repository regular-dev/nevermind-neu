use std::option::Option;

use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;

// not used
pub struct DummyLayer {
    output: Blob,
}

impl AbstractLayer for DummyLayer {
    fn forward(&mut self, input: &Blob) -> &Blob
    {
        &self.output
    }
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob)
    {
        (&self.output, &self.output)
    }

    fn optimize(&mut self, _prev_out: &Blob) -> &Blob {
        &self.output
    }

    fn layer_name(&self) -> &str {
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