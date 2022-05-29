use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;

use std::mem::replace;
use std::vec::Vec;

pub struct InputDataLayer {
    pub next_layers: Vec< Box< dyn AbstractLayer > >,
    pub cur_data: Option<Blob>,
    pub input_size: i32
}

impl AbstractLayer for InputDataLayer {
    fn forward(&mut self, input: Blob) -> Blob
    {
        if self.cur_data.is_none() {
            return input.clone();
        }

        let s = std::mem::replace(&mut self.cur_data, None);
        return s.unwrap();
    }
    fn backward(&mut self, input: Blob) -> Blob
    {
        // dummy
        return Blob::new();
    }

    fn layer_name(&mut self) -> &str
    {
        "InputDataLayer"
    }

    fn next_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        return Some(&mut self.next_layers[idx]);
    }
    fn previous_layer(& mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        None
    }

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >)
    {
        self.next_layers.push(layer);
    }
}

impl InputDataLayer {
    pub fn load_data(&mut self, input: Blob) 
    {
        self.cur_data = Some(input);
    }

    pub fn new(input_size: i32) -> Self {
        Self {
            next_layers: Vec::new(),
            cur_data: None,
            input_size
        }
    }
}