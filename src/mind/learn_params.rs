use uuid::Uuid;

use std::cell::RefCell;
use std::rc::Rc;

use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::util::{Blob, DataVec, WsBlob, WsMat};

#[derive(Clone)]
pub struct LearnParams {
    pub ws: Rc<RefCell<WsBlob>>,
    pub ws_grad: Rc<RefCell<WsBlob>>,
    pub err_vals: Rc<RefCell<DataVec>>,
    pub output: Rc<RefCell<DataVec>>,
    pub uuid: Uuid,
}

pub type LearnParamsPtr = Rc<RefCell<LearnParams>>;
pub type ParamsBlob = Vec<LearnParams>;

impl LearnParams {
    pub fn new(size: usize, prev_size: usize) -> Self {
        Self {
            ws: Rc::new(RefCell::new(vec![WsMat::random(
                (size, prev_size),
                Uniform::new(-0.5, 0.5),
            )])),
            ws_grad: Rc::new(RefCell::new(vec![WsMat::zeros((size, prev_size))])),
            err_vals: Rc::new(RefCell::new(DataVec::zeros(size))),
            output: Rc::new(RefCell::new(DataVec::zeros(size))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn new_with_const_bias(size: usize, prev_size: usize) -> Self {
        let ws = WsMat::random((size, prev_size), Uniform::new(-0.5, 0.5));
        let ws_bias = WsMat::random((size, 1), Uniform::new(-0.5, 0.5));

        Self {
            ws: Rc::new(RefCell::new(vec![ws, ws_bias])),
            ws_grad: Rc::new(RefCell::new(vec![
                WsMat::zeros((size, prev_size)),
                WsMat::zeros((size, 1)),
            ])),
            err_vals: Rc::new(RefCell::new(DataVec::zeros(size))),
            output: Rc::new(RefCell::new(DataVec::zeros(size))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn new_only_output(size: usize) -> Self {
        Self {
            ws: Rc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            ws_grad: Rc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            err_vals: Rc::new(RefCell::new(DataVec::zeros(0))),
            output: Rc::new(RefCell::new(DataVec::zeros(size))),
            uuid: Uuid::new_v4(),
        }
    }
}
