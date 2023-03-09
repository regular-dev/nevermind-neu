use crate::layers::*;
use crate::learn_params::LearnParams;
use crate::util::*;

use ocl::{Context, Device, Kernel, Program, Queue};

use std::collections::HashMap;

static EUCLIDEAN_LOSS_KERNEL: &'static str = r#"
    float sigmoid(float v) 
    {
        return 1.0 / (1.0 + exp(-v)); 
    }

    __kernel void euclidean_loss(
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

pub struct EuclideanLossLayerOcl {
    cpu_params: LearnParams,
    gpu_params: OclParams,
    size: usize,
    ocl_queue: Option<Queue>,
    ocl_kernel: Option<Kernel>,
}

impl EuclideanLossLayerOcl {
    pub fn new(size: usize) -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            gpu_params: OclParams::empty(),
            size,
            ocl_queue: None,
            ocl_kernel: None,
        }
    }
}

impl AbstractLayer for EuclideanLossLayerOcl {
    fn layer_type(&self) -> &str {
        "EuclideanLossLayerOcl"
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
        init_ocl_params(&mut self.gpu_params, queue.clone(), self.size, sh)
            .expect("Buffer create failure");
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

impl AbstractLayerOcl for EuclideanLossLayerOcl {
    fn init_ocl(
        &mut self,
        ocl_ctx: &Context,
        device: Device,
        queue: Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let program = Program::builder()
            .devices(device)
            .src(EUCLIDEAN_LOSS_KERNEL)
            .build(&ocl_ctx)?;

        let kern = Kernel::builder()
            .name("euclidean_loss")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(self.size)
            .build()?;

        self.ocl_queue = Some(queue);
        self.ocl_kernel = Some(kern);

        Ok(())
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

impl Default for EuclideanLossLayerOcl {
    fn default() -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            gpu_params: OclParams::empty(),
            size: 0,
            ocl_queue: None,
            ocl_kernel: None,
        }
    }
}

impl Clone for EuclideanLossLayerOcl {
    fn clone(&self) -> Self {
        let queue = self.ocl_queue.as_ref().unwrap();

        Self {
            cpu_params: self.cpu_params.clone(),
            gpu_params: self.gpu_params.clone(),
            size: self.size,
            ocl_kernel: None,
            ocl_queue: Some(queue.clone()),
        }
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}
