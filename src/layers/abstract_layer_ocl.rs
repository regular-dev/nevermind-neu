use std::{cell::RefCell, error::Error, rc::Rc};

use ocl::{Buffer, Context, Device, MemFlags, ProQue, Queue};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::layers::*;
use crate::util::*;

pub type LayerOclResult = Result<Vec<OclParams>, LayerError>;

#[derive(Clone)]
pub struct OclParams {
    pub output: Rc<RefCell<Buffer<Num>>>,
    pub ws: Rc<RefCell<Buffer<Num>>>,
    pub neu_grad: Rc<RefCell<Buffer<Num>>>,
    pub ws_grad: Rc<RefCell<Buffer<Num>>>,
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
}

pub trait AbstractLayerOcl: AbstractLayer {
    fn init_ocl(
        &mut self,
        ocl_ctx: &Context,
        device: Device,
        queue: Queue,
    ) -> Result<(), Box<dyn Error>> {
        todo!("ocl")
    }

    fn forward_input_ocl(&mut self, input_data: Batch) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }

    fn forward_ocl(&mut self, params: OclParams) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }
    fn backward_ocl(
        &mut self,
        prev_input: OclParams,
        next_input: OclParams,
    ) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }

    fn ocl_params(&self) -> Option<OclParams>;

    // Do copy layer memory(ws, output, ...)
    fn copy_layer_ocl(&self) -> Box<dyn AbstractLayerOcl>;

    // Do copy only Rc
    fn clone_layer_ocl(&self) -> Box<dyn AbstractLayerOcl>;
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

        let ws_cpu_vals = WsMat::random((self_size, prev_shape[0]), Uniform::new(-0.9, 0.9));

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
            ws_grad: Rc::new(RefCell::new(ws_grad))
        };

        Ok(params)
}
