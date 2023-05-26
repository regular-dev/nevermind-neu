use std::{cell::RefCell, error::Error, rc::Rc};

use uuid::Uuid;

use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ocl::{Buffer, Context, Device, MemFlags, ProQue, Queue};

use crate::layers::*;
use crate::models::pb::*;
use crate::util::*;

pub type LayerOclResult = Result<Vec<OclParams>, LayerError>;

#[derive(Clone)]
pub struct OclParams {
    pub output: Rc<RefCell<Buffer<Num>>>,
    pub ws: Rc<RefCell<Buffer<Num>>>,
    pub neu_grad: Rc<RefCell<Buffer<Num>>>,
    pub ws_grad: Rc<RefCell<Buffer<Num>>>,
    pub ws_shape: [usize; 2],
    pub uuid: Uuid,
}

pub type OclParamsBlob = Vec<OclParams>;

pub enum FetchParams {
    Output,
    Ws,
    NeuGrad,
    WsGrad,
    OutputAndNeuGrad,
    All,
}

impl OclParams {
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
            ws_shape: [0, 0],
            uuid: Uuid::new_v4(),
        }
    }

    pub fn serialize_ws_to_pb(&self) -> PbWsBlob {
        let mut pb_ws = PbWsBlob::default();
        let ws_b = self.ws.borrow();
        let mut vec_ws = vec![0.0; ws_b.len()];

        ws_b.read(&mut vec_ws)
            .enq()
            .expect("Failed to serialize weights");

        let pb_vec = PbFloatVec {
            vals: vec_ws,
            shape_size: self.ws_shape[0] as i32,
            shape_prev_size: self.ws_shape[1] as i32,
        };

        pb_ws.ws.push(pb_vec);

        pb_ws
    }

    pub fn set_ws_from_vec(&mut self, v: &mut Vec<f32>, q: Queue) {
        let ws = Buffer::builder()
            .queue(q)
            .flags(MemFlags::new().read_write())
            .len(self.ws_shape[0] * self.ws_shape[1])
            .copy_host_slice(v.as_slice())
            .build()
            .expect("Failed to create ws buffer from vec");

        self.ws = Rc::new(RefCell::new(ws));
    }
}

pub fn init_ocl_params(
    queue: Queue,
    self_size: usize,
    prev_shape: &[usize],
) -> Result<OclParams, Box<dyn Error>> {
    let output = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size)
        .build()?;
    #[cfg(feature = "opencl")]
    let neu_grad = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size)
        .build()?;
    let ws_grad = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size * prev_shape[0])
        .build()?;

    let ws_cpu_vals = WsMat::random((self_size, prev_shape[0]), Uniform::new(-0.1, 0.1));

    let ws = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size * prev_shape[0])
        .copy_host_slice(ws_cpu_vals.as_slice().unwrap())
        .build()?;

    let params = OclParams {
        output: Rc::new(RefCell::new(output)),
        ws: Rc::new(RefCell::new(ws)),
        neu_grad: Rc::new(RefCell::new(neu_grad)),
        ws_grad: Rc::new(RefCell::new(ws_grad)),
        ws_shape: [self_size, prev_shape[0]],
        uuid: Uuid::new_v4(),
    };

    Ok(params)
}

pub fn fit_to_batch_size_ocl(
    params: OclParams,
    self_size: usize,
    batch_size: usize,
    queue: Queue,
) -> Result<OclParams, Box<dyn Error>> {
    let output = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size * batch_size)
        .build()?;
    let neu_grad = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size * batch_size)
        .build()?;

    let mut output_b = params.output.borrow_mut();
    let mut neu_grad_b = params.neu_grad.borrow_mut();

    *output_b = output;
    *neu_grad_b = neu_grad;

    drop(output_b);
    drop(neu_grad_b);

    Ok(params)
}
