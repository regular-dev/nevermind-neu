use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;
use crate::mind::abstract_layer::DataVec;

use super::activation::sigmoid_on_vec;

use std::mem::replace;
use std::vec::Vec;

pub struct InputDataLayer {
    pub next_layers: Vec< Box< dyn AbstractLayer > >,
    pub input_size: usize,
    pub output: Blob,
}

impl AbstractLayer for InputDataLayer {
    fn forward(&mut self, input: &Blob) -> &Blob
    {
        let in_vec = &input[0];
        let out_vec = &mut self.output[0];

       sigmoid_on_vec(in_vec, out_vec);
        
        &self.output
    }
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob)
    {
        (&self.output, &self.output)
    }

    fn optimize(&mut self, _prev_out: &Blob) -> &Blob {
        &self.output
    }

    fn layer_name(&self) -> &str
    {
        "InputDataLayer"
    }

    fn size(&self) -> usize {
        self.input_size
    }
}

impl InputDataLayer {
    pub fn load_data(&mut self, input: Blob) 
    {
    }

    pub fn new(input_size: usize) -> Self {
        let mut vec_outp = DataVec::new();
        vec_outp.resize(input_size, 0.0);

        Self {
            next_layers: Vec::new(),
            input_size,
            output: vec![vec_outp]
        }
    }
}