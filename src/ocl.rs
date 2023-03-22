use std::{cell::RefCell, error::Error, rc::Rc};

use uuid::Uuid;

use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ocl::{Buffer, Context, Device, MemFlags, ProQue, Queue};

use crate::layers::*;
use crate::util::*;

pub type LayerOclResult = Result<Vec<OclParams>, LayerError>;

#[derive(Clone)]
pub struct OclParams {
    pub output: Rc<RefCell<Buffer<Num>>>,
    pub ws: Rc<RefCell<Buffer<Num>>>,
    pub neu_grad: Rc<RefCell<Buffer<Num>>>,
    pub ws_grad: Rc<RefCell<Buffer<Num>>>,
    pub uuid: Uuid,
}

pub enum FetchParams {
    Output,
    Ws,
    NeuGrad,
    WsGrad,
    OutputAndNeuGrad,
    All,
}

impl OclParams {
    // pub fn empty() -> Self {
    //     Self {
    //         output: Rc::new(RefCell::new(Buffer::builder().build().unwrap())),
    //         ws: Rc::new(RefCell::new(Buffer::builder().build().unwrap())),
    //         neu_grad: Rc::new(RefCell::new(Buffer::builder().build().unwrap())),
    //         ws_grad: Rc::new(RefCell::new(Buffer::builder().build().unwrap())),
    //     }
    // }

    pub fn only_output(buf: Buffer<f32>, queue: Queue) -> Self {
        Self {
            output: Rc::new(RefCell::new(buf)),
            ws: Rc::new(RefCell::new(
                Buffer::builder()
                    .queue(queue.clone())
                    .len(1)
                    .build()
                    .unwrap(),
            )),
            neu_grad: Rc::new(RefCell::new(
                Buffer::builder()
                    .queue(queue.clone())
                    .len(1)
                    .build()
                    .unwrap(),
            )),
            ws_grad: Rc::new(RefCell::new(
                Buffer::builder().queue(queue).len(1).build().unwrap(),
            )),
            uuid: Uuid::new_v4(),
        }
    }
}

pub type OclParamsBlob = Vec<OclParams>;