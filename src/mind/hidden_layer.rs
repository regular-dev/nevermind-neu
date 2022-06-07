use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;
use crate::mind::activation;

use rand::Rng;

use super::activation::sigmoid_prime;

pub struct HiddenLayer {
    pub ws: Blob,
    pub next_layer: Option<Box<dyn AbstractLayer>>,
    pub prev_layer: Option<Box<dyn AbstractLayer>>,
    pub neu_count: usize,
    pub output: Blob,
    pub err_vals: Blob,
}

impl AbstractLayer for HiddenLayer {
    fn forward(&mut self, input: &Blob) -> &Blob {
        let inp_vec = &input[0];
        let out_vec = &mut self.output[0];

        for idx_out in 0..out_vec.len() {
            let mut sum: f32 = 0.0;
            for (idx_in, val_in) in inp_vec.iter().enumerate() {
                sum += val_in * self.ws[0][idx_out * inp_vec.len() + idx_in];
            }
            let mut activated_val = activation::sigmoid(sum);
            out_vec[idx_out] = activated_val;
        }

        &self.output
    }
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob) {
        let inp_vec = &input[0];
        let ws_vec = &weights[0];
        let err_vec = &mut self.err_vals[0];

        for idx in 0..self.neu_count {
            let mut sum: f32 = 0.0;
            for inp_idx in 0..inp_vec.len() {
                sum += inp_vec[inp_idx] * ws_vec[inp_idx*inp_vec.len()+idx];
            }

            err_vec[idx] = sigmoid_prime(self.output[0][idx] * sum);
        }

        (&self.err_vals, &self.ws)
    }

    fn layer_name(&self) -> &str {
        "HiddenLayer"
    }

    fn next_layer(&mut self, _idx: usize) -> Option<&mut Box<dyn AbstractLayer>> {
        if let Some(l) = &mut self.next_layer {
            return Some(l);
        } else {
            return None;
        }
    }

    fn previous_layer(&mut self, idx: usize) -> Option<&mut Box<dyn AbstractLayer>> {
        if let Some(l) = &mut self.prev_layer {
            return Some(l);
        } else {
            return None;
        }
    }

    fn add_next_layer(&mut self, layer: Box<dyn AbstractLayer>) {
        self.next_layer = Some(layer);
    }

    fn size(&self) -> usize {
        self.neu_count
    }
}

impl HiddenLayer {
    pub fn new(size: usize, prev_size: usize) -> Self {
        let mut ws: Vec<Vec<f32>> = Vec::new();
        ws.resize(1, Vec::new());

        for i in &mut ws {
            i.resize_with(prev_size * size, || {
                rand::thread_rng().gen_range(0.01, 0.55)
            });
        }

        let mut out_vec = Vec::new();
        out_vec.resize(size, 0.0);

        let err_vec = out_vec.clone();

        Self {
            neu_count: size,
            ws,
            next_layer: None,
            prev_layer: None,
            output: vec![out_vec],
            err_vals: vec![err_vec]
        }
    }
}
