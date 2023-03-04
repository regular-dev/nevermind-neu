use std::{cell::RefCell, rc::Rc};

use ocl::{Buffer, MemFlags, ProQue};

use crate::util::*;

use super::{AbstractLayer, LayerForwardResult, LayerBackwardResult};

pub struct OclParams {
    output: Rc<RefCell<Buffer<Num>>>,
    ws: Rc<RefCell<Buffer<Num>>>,
    neu_grad: Rc<RefCell<Buffer<Num>>>,
    ws_grad: Rc<RefCell<Buffer<Num>>>,
}

pub trait AbstractLayerOcl : AbstractLayer {
    fn forward_ocl(&mut self, params: OclParams) -> LayerForwardResult;
    fn backward_ocl(&mut self, prev_input: OclParams, next_input: OclParams) -> LayerBackwardResult;

    fn ocl_params(&self) -> Option<OclParams>;
}