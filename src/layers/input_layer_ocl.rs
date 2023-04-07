use ocl::{Buffer, Context, Device, MemFlags, Queue};

use log::debug;

use crate::layers::*;
use crate::learn_params::LearnParams;
use crate::ocl::*;
use crate::util::*;

use std::{collections::HashMap, error::Error};

#[derive(Clone)]
pub struct InputLayerOcl {
    ocl_params: Option<OclParams>,
    size: usize,
    batch_size: usize,
    ocl_queue: Option<Queue>,
}

impl InputLayerOcl {
    pub fn new(size: usize) -> Self {
        Self {
            ocl_params: None,
            size,
            batch_size: 1,
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

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    fn learn_params(&self) -> Option<LearnParams> {
        None
    }

    fn set_learn_params(&mut self, lp: LearnParams) {}

    fn set_input_shape(&mut self, sh: &[usize]) {}

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

    fn forward_input_ocl(&mut self, input_data: Array2D) -> LayerOclResult {
        let ocl_queue = self.ocl_queue.as_ref().unwrap();
        let ocl_buf = Buffer::builder()
            .queue(ocl_queue.clone())
            .flags(MemFlags::new().read_write())
            .len(input_data.len())
            .copy_host_slice(input_data.as_slice().unwrap())
            .build()
            .expect("[inp_ocl] Couldn't create "); // TODO : handle unwrap

        self.ocl_params = Some(OclParams::only_output(ocl_buf, ocl_queue.clone()));

        Ok(vec![self.ocl_params.as_ref().unwrap().clone()])
    }

    fn forward_ocl(&mut self, params: OclParamsBlob) -> LayerOclResult {
        Err(LayerError::NotImpl)
    }

    fn ocl_params(&self) -> Option<OclParams> {
        Some(self.ocl_params.as_ref().unwrap().clone())
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
            ocl_params: None,
            size: 0,
            batch_size: 1,
            ocl_queue: None,
        }
    }
}

impl WithParams for InputLayerOcl {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut out = HashMap::new();

        out.insert("size".to_string(), Variant::Int(self.size as i32));

        out
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if let Some(size) = args.get("size") {
            if let Variant::Int(size) = size {
                self.size = *size as usize;
            }
        }
    }
}
