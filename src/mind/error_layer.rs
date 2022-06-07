use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;
use crate::mind::abstract_layer::DataVec;
use crate::mind::activation::{sigmoid, sigmoid_prime};

use rand::Rng;

pub struct ErrorLayer {
    pub next_layer: Option< Box< dyn AbstractLayer > >,
    pub error: f32,
    pub error_vals: Blob,
    pub ws: Blob,
    pub size: usize,
    pub output: Blob,
}

impl AbstractLayer for ErrorLayer {
    fn forward(&mut self, input: &Blob) -> &Blob
    {
        let inp_vec = &input[0];
        let out_vec = &mut self.output[0];

        for out_idx in 0..out_vec.len() {
            let mut sum: f32 = 0.0;

            for (inp_idx, inp_iter) in inp_vec.iter().enumerate() {
                sum += inp_iter * self.ws[0][out_idx*inp_vec.len()+inp_idx];
            }

            let activated = sigmoid(sum);
            out_vec[out_idx] = activated;
        }

        &self.output
    }
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob)
    {
        let expected_vec = &input[0];
        let out_vec = &self.output[0];
        let err_vec = &mut self.error_vals[0];
        
        for (idx, _out_iter) in out_vec.iter().enumerate() {
            err_vec[idx] = (expected_vec[idx] - out_vec[idx]) * sigmoid_prime(out_vec[idx]);
        }

        (&self.error_vals, &self.ws)
    }

    fn layer_name(&self) -> &str
    {
        "ErrorLayer"
    }

    fn next_layer(&mut self, _idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        if let Some(l) = &mut self.next_layer {
            return Some(l);
        } else {
            return None;
        }
    }
    fn previous_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        None
    }

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >)
    {

    }

    fn size(&self) -> usize 
    {
        self.size
    }
}

impl ErrorLayer {
    pub fn new(size: usize, prev_size: usize) -> Self 
    {
        let mut out_vec: Vec<f32> = Vec::new();
        out_vec.resize(size, 0.0);

        let err_vals = out_vec.clone();

        let mut ws: Vec< Vec< f32 > > = Vec::new();
        ws.resize(1, Vec::new());

        for i in &mut ws {
            i.resize_with(prev_size*size, || { rand::thread_rng().gen_range(0.01, 0.55) } );
        }


        Self {
            size,
            next_layer: None,
            error: 0.0,
            output: vec![out_vec],
            ws: ws,
            error_vals: vec![err_vals]
        }
    }
}