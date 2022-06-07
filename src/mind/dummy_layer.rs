use std::option::Option;

use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;

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

    fn next_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        return None;
    }
    fn previous_layer(&mut self, idx: usize) -> Option < &mut Box< dyn AbstractLayer > >
    {
        return None;
    }

    fn layer_name(&self) -> &str {
        return "DummyLayer";
    }

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >)
    {

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