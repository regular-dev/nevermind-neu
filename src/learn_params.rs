use uuid::Uuid;

use std::cell::RefCell;
use std::rc::Rc;
use std::ops::DerefMut;

use ndarray_rand::rand_distr::{Uniform, Distribution};
use ndarray_rand::rand::{SeedableRng};
use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::RandomExt;

use super::util::{DataVec, WsBlob, WsMat, Batch};

#[derive(Clone, Default)]
pub struct LearnParams {
    pub ws: Rc<RefCell<WsBlob>>,
    pub ws_grad: Rc<RefCell<WsBlob>>,
    pub err_vals: Rc<RefCell<Batch>>,
    pub output: Rc<RefCell<Batch>>,
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
            err_vals: Rc::new(RefCell::new(Batch::zeros((1, size)))),
            output: Rc::new(RefCell::new(Batch::zeros((1, size)))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn new_with_const_bias(size: usize, prev_size: usize) -> Self {
        let ws = WsMat::random((size, prev_size), Uniform::new(-0.5, 0.5));
        // let ws_bias = WsMat::random((size, 1), Uniform::new(-0.5, 0.5));
        let ws_bias = WsMat::from_elem((size, 1), 0.1);

        Self {
            ws: Rc::new(RefCell::new(vec![ws, ws_bias])),
            ws_grad: Rc::new(RefCell::new(vec![
                WsMat::zeros((size, prev_size)),
                WsMat::zeros((size, 1)),
            ])),
            err_vals: Rc::new(RefCell::new(Batch::zeros((1, size)))),
            output: Rc::new(RefCell::new(Batch::zeros((1, size)))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn new_only_output(size: usize) -> Self {
        Self {
            ws: Rc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            ws_grad: Rc::new(RefCell::new(vec![WsMat::zeros((0, 0))])),
            err_vals: Rc::new(RefCell::new(Batch::zeros((0, 0)))),
            output: Rc::new(RefCell::new(Batch::zeros((0, 0)))),
            uuid: Uuid::new_v4(),
        }
    }

    pub fn fit_to_batch_size(&mut self, new_batch_size: usize) {
        let mut out_m = self.output.borrow_mut();    
        let mut err_m = self.err_vals.borrow_mut();
        let size = out_m.ncols();

        if out_m.nrows() != new_batch_size {
            *out_m = Batch::zeros((new_batch_size, size));
            *err_m = Batch::zeros((new_batch_size, size));
        }
    }

    pub fn drop_ws(&mut self, dropout: f32) {
        let mut ws = self.ws.borrow_mut();
        let between = Uniform::from((0.0)..(1.0));
        let mut rng = SmallRng::from_entropy();

        for i in ws.deref_mut() {
            for it_ws in i.iter_mut() {
                let n = between.sample(&mut rng);

                if n > dropout {
                    *it_ws = 0.0;
                }
            }
        }
    }
}
