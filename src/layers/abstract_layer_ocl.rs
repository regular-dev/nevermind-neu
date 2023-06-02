use std::{error::Error};

use ocl::{Buffer, Context, Device, MemFlags, ProQue, Queue};

use crate::ocl::*;
use crate::layers::*;
use crate::util::*;

pub trait AbstractLayerOcl: AbstractLayer {
    fn init_ocl(
        &mut self,
        _ocl_ctx: &Context,
        _device: Device,
        _queue: Queue,
    ) -> Result<(), Box<dyn Error>>;

    fn forward_input_ocl(&mut self, _input_data: Array2D) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }
    fn forward_ocl(&mut self, _params: OclParamsBlob) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }
    fn backward_ocl(
        &mut self,
        _prev_input: OclParamsBlob,
        _next_input: OclParamsBlob,
    ) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }
    fn backward_output_ocl(
        &mut self,
        _prev_input: OclParamsBlob,
        _expected: Array2D,
    ) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }

    fn fetch_params_to_cpu(&self, _t: FetchParams) { todo!() }

    fn ocl_params(&self) -> Option<OclParams>;
    fn set_ocl_params(&mut self, _params: OclParams) {}

    // Do copy layer memory(ws, output, ...)
    fn copy_layer_ocl(&self) -> Box<dyn AbstractLayerOcl>;

    // Do copy only Rc
    fn clone_layer_ocl(&self) -> Box<dyn AbstractLayerOcl>;
}