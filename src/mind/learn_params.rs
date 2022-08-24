use uuid::Uuid;

use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::util::{WsBlob, Blob, WsMat, DataVec};


pub struct LearnParams {
    pub ws: WsBlob,
    pub ws_grad: WsBlob,
    pub err_vals: DataVec,
    pub output: DataVec,
    pub uuid: Uuid,
}

impl LearnParams {
    pub fn new(size: usize, prev_size: usize) -> Self {
        Self {
            ws: vec![WsMat::random((size, prev_size), Uniform::new(-0.5, 0.5))],
            ws_grad: vec![WsMat::zeros((size, prev_size))],
            err_vals: DataVec::zeros(size),
            output: DataVec::zeros(size),
            uuid: Uuid::new_v4()
        }
    }

    pub fn new_with_const_bias(size: usize, prev_size: usize) -> Self {
        let ws = WsMat::random((size, prev_size), Uniform::new(-0.5, 0.5));
        let ws_bias = WsMat::random( (size, 1), Uniform::new(-0.5, 0.5));

        Self {
            ws: vec![ws, ws_bias],
            ws_grad: vec![WsMat::zeros((size, prev_size))],
            err_vals: DataVec::zeros(size),
            output: DataVec::zeros(size),
            uuid: Uuid::new_v4()
        }
    }

    pub fn new_only_output(size: usize) -> Self {
        Self {
            ws: vec![WsMat::zeros((0, 0))],
            ws_grad: vec![WsMat::zeros((0, 0))],
            err_vals: DataVec::zeros(0),
            output: DataVec::zeros(size),
            uuid: Uuid::new_v4()
        }
    }
}