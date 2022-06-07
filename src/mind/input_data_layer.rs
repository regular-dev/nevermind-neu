use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;
use crate::mind::abstract_layer::DataVec;

use super::activation::sigmoid_on_vec;

use std::mem::replace;
use std::vec::Vec;

pub struct InputDataLayer {
    pub next_layers: Vec< Box< dyn AbstractLayer > >,
    pub cur_data: Option<Blob>,
    pub input_size: usize,
    pub output: Blob,
}

impl AbstractLayer for InputDataLayer {
    fn forward(&mut self, input: &Blob) -> &Blob
    {
        if let Some(d) = &self.cur_data {
            let in_vec = &d[0];
            let out_vec = &mut self.output[0];
            sigmoid_on_vec(in_vec,out_vec);
        }
        
        &self.output
    }
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob)
    {
        // dummy
        (&self.output, &self.output)
    }

    fn layer_name(&self) -> &str
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

    fn size(&self) -> usize {
        self.input_size
    }
}

impl InputDataLayer {
    pub fn load_data(&mut self, input: Blob) 
    {
        self.cur_data = Some(input);
    }

    pub fn new(input_size: usize) -> Self {
        let mut vec_outp = DataVec::new();
        vec_outp.resize(input_size, 0.0);

        Self {
            next_layers: Vec::new(),
            cur_data: None,
            input_size,
            output: vec![vec_outp]
        }
    }
}