use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;
use crate::mind::activation::{sigmoid, sigmoid_deriv};

use rand::Rng;

pub struct HiddenLayer {
    pub ws: Blob,
    pub ws_delta: Blob,
    pub size: usize,
    pub output: Blob,
    pub err_vals: Blob,
    pub prev_size: usize,
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
            let activated_val = sigmoid(sum);
            out_vec[idx_out] = activated_val;
        }

        &self.output
    }
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob) {
        let inp_vec = &input[0];
        let ws_vec = &weights[0];
        let err_vec = &mut self.err_vals[0];

        for idx in 0..self.size {
            let mut sum: f32 = 0.0;
            for inp_idx in 0..inp_vec.len() {
                sum += inp_vec[inp_idx] * ws_vec[inp_idx*self.size+idx];
            }

            err_vec[idx] = sigmoid_deriv(self.output[0][idx] ) * sum;
        }

        (&self.err_vals, &self.ws)
    }

    fn optimize(&mut self, prev_out: &Blob) -> &Blob {
        let prev_vec = &prev_out[0];

        for neu_idx in 0..self.size {
            for prev_idx in 0..self.prev_size {
                let ws_idx = neu_idx * self.prev_size + prev_idx;

                // 0.2 - ALPHA
                self.ws[0][ws_idx] += 0.2 * self.ws_delta[0][ws_idx];
                // 0.2 - LEARNING RATE
                self.ws_delta[0][ws_idx] = 0.2 * self.err_vals[0][neu_idx] * prev_vec[prev_idx];

                self.ws[0][ws_idx] += self.ws_delta[0][ws_idx];
            }
        }

        &self.output
    }

    fn layer_name(&self) -> &str {
        "HiddenLayer"
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl HiddenLayer {
    pub fn new(size: usize, prev_size: usize) -> Self {
        let mut ws: Vec<Vec<f32>> = Vec::new();
        ws.resize(1, Vec::new());

        for i in &mut ws {
            i.resize_with(prev_size * size, || {
                rand::thread_rng().gen_range(-0.55, 0.55)
            });
        }

        let mut ws_delta = Vec::new();
        ws_delta.resize(prev_size*size, 0.0);

        let mut out_vec = Vec::new();
        out_vec.resize(size, 0.0);

        let err_vec = out_vec.clone();

        Self {
            size,
            prev_size,
            ws,
            ws_delta: vec![ws_delta],
            output: vec![out_vec],
            err_vals: vec![err_vec]
        }
    }
}
