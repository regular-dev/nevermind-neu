use std::option::Option;

use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;

pub struct DummyLayer {

}

impl AbstractLayer for DummyLayer {
    fn forward(&mut self, input: Blob) -> Blob
    {
        Blob::new()
    }
    fn backward(&mut self, input: Blob) -> Blob
    {
        Blob::new()
    }

    fn next_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        return None;
    }
    fn previous_layer(&mut self, idx: usize) -> Option < &mut Box< dyn AbstractLayer > >
    {
        return None;
    }

    fn layer_name(&mut self) -> &str {
        return "DummyLayer";
    }

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >)
    {

    }
}

impl DummyLayer {
    pub fn new() -> Self {
        DummyLayer{}
    }
}