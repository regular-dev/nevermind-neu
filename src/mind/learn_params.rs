use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::util::{WsBlob, Blob, WsMat, DataVec};


pub struct LearnParams {
    pub ws: WsBlob,
    pub ws_delta: WsBlob,
    pub err_vals: Blob,
    pub output: Blob,
}

impl LearnParams {
    pub fn new(size: usize, prev_size: usize) -> Self {
        Self {
            ws: vec![WsMat::random((size, prev_size), Uniform::new(-0.5, 0.5))],
            ws_delta: vec![WsMat::zeros((size, prev_size))],
            err_vals: vec![DataVec::zeros(size)],
            output: vec![DataVec::zeros(size)]
        }
    }

    pub fn new_only_output(size: usize) -> Self {
        Self {
            ws: vec![WsMat::zeros((0, 0))],
            ws_delta: vec![WsMat::zeros((0, 0))],
            err_vals: vec![DataVec::zeros(0)],
            output: vec![DataVec::zeros(size)]
        }
    }
}