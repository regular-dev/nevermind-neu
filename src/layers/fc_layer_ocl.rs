use crate::layers::*;
use crate::learn_params::LearnParams;
use crate::util::*;

use ndarray_rand::{rand_distr::Uniform, RandomExt};

use ocl::{Buffer, Context, Device, Kernel, MemFlags, Program, Queue};

use std::{collections::HashMap, error::Error};

static FC_LAYER_KERNEL: &'static str = r#"
    float sigmoid(float v) 
    {
        return 1.0 / (1.0 + exp(-v)); 
    }

    __kernel void fc_layer_product(
                __global const float *x,
                __global const float *ws,
                __global float *y)
    {
        const int first_layer = 1 << 12;
        const int out_layer = 1 << 14;
        uint const idx = get_global_id(0);

        __private float sum = 0.0;

        for (int j = 0; j < first_layer; ++j) {
            sum += ws[idx * first_layer + j] *  x[j];
        }
            
        y[idx] = sigmoid(sum);
    }
"#;

//#[derive(Clone)]
pub struct FcLayerOcl {
    cpu_params: LearnParams,
    ocl_params: Option<OclParams>,
    size: usize,

    ocl_kernel: Option<Kernel>,
    ocl_queue: Option<Queue>,
}

impl FcLayerOcl {
    pub fn new(size: usize) -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            ocl_params: None,
            size,
            ocl_kernel: None,
            ocl_queue: None,
        }
    }
}

impl AbstractLayer for FcLayerOcl {
    fn layer_type(&self) -> &str {
        "FcLayerOcl"
    }

    fn size(&self) -> usize {
        self.size
    }

    fn learn_params(&self) -> Option<LearnParams> {
        None
    }

    fn set_learn_params(&mut self, lp: LearnParams) {}

    fn set_input_shape(&mut self, sh: &[usize]) {
        let queue = self.ocl_queue.as_ref().unwrap();
        // buffer routine
        self.ocl_params = Some(init_ocl_params(queue.clone(), self.size, sh)
            .expect("Buffer create failure"));
    }

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

impl AbstractLayerOcl for FcLayerOcl {
    fn init_ocl(
        &mut self,
        ocl_ctx: &Context,
        device: Device,
        queue: Queue,
    ) -> Result<(), Box<dyn Error>> {
        let program = Program::builder()
            .devices(device)
            .src(FC_LAYER_KERNEL)
            .build(&ocl_ctx)?;

        let kern = Kernel::builder()
            .name("fc_layer_product")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(self.size)
            .arg(None::<&Buffer<f32>>)
            .arg(None::<&Buffer<f32>>)
            .arg(None::<&Buffer<f32>>)
            .build()?;

        self.ocl_kernel = Some(kern);
        self.ocl_queue = Some(queue);

        Ok(())
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

impl Default for FcLayerOcl {
    fn default() -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            ocl_params: None,
            size: 0,
            ocl_kernel: None,
            ocl_queue: None,
        }
    }
}

impl Clone for FcLayerOcl {
    fn clone(&self) -> Self {
        let queue = self.ocl_queue.as_ref().unwrap();

        Self {
            cpu_params: self.cpu_params.clone(),
            ocl_params: self.ocl_params.clone(),
            size: self.size,
            ocl_kernel: None,
            ocl_queue: Some(queue.clone()),
        }
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}
