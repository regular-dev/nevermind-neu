use ocl::{Buffer, Context, Device, MemFlags, Queue};

use crate::layers::*;
use crate::learn_params::LearnParams;
use crate::util::*;

use std::{collections::HashMap, error::Error};

#[derive(Clone)]
pub struct InputLayerOcl {
    gpu_params: OclParams,
    size: usize,
    ocl_queue: Option<Queue>,
}

impl InputLayerOcl {
    pub fn new(size: usize) -> Self {
        Self {
            gpu_params: OclParams::empty(),
            size,
            ocl_queue: None,
        }
    }
}

impl AbstractLayer for InputLayerOcl {
    fn layer_type(&self) -> &str {
        "InputLayerOcl"
    }

    fn size(&self) -> usize {
        self.size
    }

    fn learn_params(&self) -> Option<LearnParams> {
        None
    }

    fn set_learn_params(&mut self, lp: LearnParams) {}

    fn set_input_shape(&mut self, sh: &[usize]) {}

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let cfg: HashMap<String, Variant> = HashMap::new();
        cfg
    }

    fn set_layer_cfg(&mut self, _cfg: &HashMap<String, Variant>) {}

    // Do copy layer memory(ws, output, ...)
    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        panic!("Do not copy OCL layers !");
    }

    // Do copy only Rc
    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        panic!("Do not copy OCL layers !");
    }
}

impl AbstractLayerOcl for InputLayerOcl {
    fn init_ocl(
        &mut self,
        _ocl_ctx: &Context,
        _device: Device,
        queue: Queue,
    ) -> Result<(), Box<dyn Error>> {
        self.ocl_queue = Some(queue);
        Ok(())
    }

    fn forward_input_ocl(&mut self, input_data: Batch) -> LayerOclResult {
        let ocl_queue = self.ocl_queue.as_ref().unwrap();

        let ocl_buf = Buffer::builder()
            .queue(ocl_queue.clone())
            .flags(MemFlags::new().read_write())
            .len(self.size)
            .copy_host_slice(input_data.as_slice().unwrap())
            .build()
            .unwrap(); // TODO : handle unwrap

        let mut inp_buf = self.gpu_params.output.borrow_mut();
        *inp_buf = ocl_buf;

        Ok(vec![self.gpu_params.clone()])
    }

    fn forward_ocl(&mut self, params: OclParams) -> LayerForwardResult {
        Err(LayerError::NotImpl)
    }

    fn backward_ocl(
        &mut self,
        prev_input: OclParams,
        next_input: OclParams,
    ) -> LayerBackwardResult {
        Err(LayerError::NotImpl)
    }

    fn ocl_params(&self) -> Option<OclParams> {
        Some(self.gpu_params.clone())
    }

    fn copy_layer_ocl(&self) -> Box<dyn AbstractLayerOcl> {
        todo!()
    }

    fn clone_layer_ocl(&self) -> Box<dyn AbstractLayerOcl> {
        Box::new(self.clone())
    }
}

impl Default for InputLayerOcl {
    fn default() -> Self {
        Self {
            gpu_params: OclParams::empty(),
            size: 0,
            ocl_queue: None
        }
    }
}
