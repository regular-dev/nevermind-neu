use std::{cell::RefCell, error::Error, rc::Rc};

use uuid::Uuid;

use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ocl::{Buffer, Context, Device, MemFlags, ProQue, Queue};

use crate::ocl::*;
use crate::layers::*;
use crate::util::*;

pub trait AbstractLayerOcl: AbstractLayer {
    fn init_ocl(
        &mut self,
        ocl_ctx: &Context,
        device: Device,
        queue: Queue,
    ) -> Result<(), Box<dyn Error>> {
        todo!("ocl")
    }

    fn forward_input_ocl(&mut self, input_data: Array2D) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }
    fn forward_ocl(&mut self, params: OclParamsBlob) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }
    fn backward_ocl(
        &mut self,
        prev_input: OclParamsBlob,
        next_input: OclParamsBlob,
    ) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }
    fn backward_output_ocl(
        &mut self,
        prev_input: OclParamsBlob,
        expected: Array2D,
    ) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }

    fn fetch_params_to_cpu(&self, t: FetchParams) { todo!() }

    fn ocl_params(&self) -> Option<OclParams>;
    fn set_ocl_params(&mut self, params: OclParams) {}

    // Do copy layer memory(ws, output, ...)
    fn copy_layer_ocl(&self) -> Box<dyn AbstractLayerOcl>;

    // Do copy only Rc
    fn clone_layer_ocl(&self) -> Box<dyn AbstractLayerOcl>;
}