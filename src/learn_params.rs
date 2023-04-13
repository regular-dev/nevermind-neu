use uuid::Uuid;

use std::cell::RefCell;
use std::ops::DerefMut;
use std::sync::Arc;

use log::debug;

use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use ndarray_rand::RandomExt;

use super::util::{Array2D, DataVec, WsBlob, WsMat};

#[derive(Clone, Default)]
pub struct LearnParams {
    pub ws: Arc<RefCell<WsBlob>>,
    pub ws_grad: Arc<RefCell<WsBlob>>,
    pub neu_grad: Arc<RefCell<Array2D>>,
    pub output: Arc<RefCell<Array2D>>,
    pub uuid: Uuid,
}

pub type LearnParamsPtr = Arc<RefCell<LearnParams>>;
pub type ParamsBlob = Vec<LearnParams>;

impl LearnParams {
    pub fn new(size: usize, prev_size: usize) -> Self {
        Self {
            ws: Arc::new(RefCell::new(vec![WsMat::random(
                (size, prev_size),
                Uniform::new(-0.9, 0.9),
            )])),
            ws_grad: Arc::new(RefCell::new(vec![WsMat::zeros((size, prev_size))])),
            neu_grad: Arc::new(RefCell::new(Array2D::zeros((1, size)))),
            output: Arc::new(RefCell::new(Array2D::zeros((1, size)))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn empty() -> Self {
        Self {
            ws: Arc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            ws_grad: Arc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            neu_grad: Arc::new(RefCell::new(Array2D::zeros((0, 0)))),
            output: Arc::new(RefCell::new(Array2D::zeros((0, 0)))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn new_only_output(size: usize) -> Self {
        Self {
            ws: Arc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            ws_grad: Arc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            neu_grad: Arc::new(RefCell::new(Array2D::zeros((0, 0)))),
            output: Arc::new(RefCell::new(Array2D::zeros((1, size)))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn new_with_const_bias(size: usize, prev_size: usize) -> Self {
        let ws = WsMat::random((size, prev_size), Uniform::new(-0.9, 0.9));
        let ws_bias = WsMat::from_elem((size, 1), 0.0);

        Self {
            ws: Arc::new(RefCell::new(vec![ws, ws_bias])),
            ws_grad: Arc::new(RefCell::new(vec![
                WsMat::zeros((size, prev_size)),
                WsMat::zeros((size, 1)),
            ])),
            neu_grad: Arc::new(RefCell::new(Array2D::zeros((1, size)))),
            output: Arc::new(RefCell::new(Array2D::zeros((1, size)))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn fit_to_batch_size(&mut self, new_batch_size: usize) {
        let mut out_m = self.output.borrow_mut();
        let mut err_m = self.neu_grad.borrow_mut();
        let size = out_m.ncols();

        if out_m.nrows() != new_batch_size {
            *out_m = Array2D::zeros((new_batch_size, size));
            *err_m = Array2D::zeros((new_batch_size, size));
        }
    }

    pub fn prepare_for_tests(&mut self, batch_size: usize) {
        let out_size = self.output.borrow().ncols();

        self.neu_grad = Arc::new(RefCell::new(Array2D::zeros((0, 0))));
        self.output = Arc::new(RefCell::new(Array2D::zeros((batch_size, out_size))));
    }

    /// Copies learn_params memory of (weights, gradients, output...)
    /// This function DO copy memory
    /// To clone only Arc<...> use .clone() function
    pub fn copy(&self) -> Self {
        let mut lp = LearnParams::default();

        {
            let mut ws_b = lp.ws.borrow_mut();
            let mut ws_grad_b = lp.ws_grad.borrow_mut();
            let mut output_b = lp.output.borrow_mut();
            let mut err_val_b = lp.neu_grad.borrow_mut();
            let uuid_b = lp.uuid.clone();

            *ws_b = self.ws.borrow().clone();
            *ws_grad_b = self.ws_grad.borrow().clone();
            *output_b = self.output.borrow().clone();
            *err_val_b = self.neu_grad.borrow().clone();
            lp.uuid = uuid_b;
        }

        lp
    }
}
