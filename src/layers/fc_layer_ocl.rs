use crate::layers::*;
use crate::learn_params::LearnParams;
use crate::util::*;

use log::debug;

use ndarray_rand::{rand_distr::Uniform, RandomExt};

use ocl::core::Int;
use ocl::{Buffer, Context, Device, Kernel, MemFlags, Program, Queue};

use std::{collections::HashMap, error::Error};

static FC_LAYER_KERNEL_FWD: &'static str = r#"
    float sigmoid(float v) 
    {
        return 1.0 / (1.0 + exp(-v)); 
    }

    __kernel void fc_layer_product(
                __private int const batch_size,
                __private int const prev_shape,
                __private int const self_shape,
                __global const float *in,
                __global const float *ws,
                __global float *out)
    {
        uint const idx = get_global_id(0);
        __private uint const real_idx = idx % self_shape;
        __private uint const batch_idx = idx / self_shape;

        __private float sum = 0.0;

        for (__private int j = 0; j < prev_shape; ++j) {
            sum += ws[real_idx * prev_shape + j] * in[j + prev_shape * batch_idx];
        }
            
        out[idx] = sigmoid(sum);
    }
"#;

static FC_LAYER_KERNEL_BWD: &'static str = r#"
    float sigmoid(float v) 
    {
        return 1.0 / (1.0 + exp(-v)); 
    }

    float sigmoid_deriv(float v) 
    {
        return (1.0 - sigmoid(v)) * sigmoid(v);
    }

    __kernel void fc_layer_grad(
                __private int const batch_size,
                __private int const prev_shape,
                __private int const next_shape,
                __private int const self_shape,
                __global const float *self_out,
                __global const float *next_grad,
                __global const float *next_ws,
                __global const float *prev_out,
                __global float *neu_grad,
                __global float *ws_grad)
    {
        uint const idx = get_global_id(0);
        __private uint const real_idx = idx % self_shape;
        __private uint const batch_idx = idx / self_shape;

        __private float sum = 0.0;

        for (__private j = 0; j < next_shape; ++j) {
            sum += next_ws[j * next_shape + real_idx] * next_grad[j + next_shape * batch_idx]
        }
       
        neu_grad[idx] = sigmoid_deriv(self_out[idx]) * sum;

        // TODO ...
    }
"#;

pub struct FcLayerOcl {
    cpu_params: LearnParams,
    ocl_params: Option<OclParams>,
    size: usize,
    batch_size: usize,

    ocl_kernel: Option<Kernel>,
    ocl_queue: Option<Queue>,
}

impl FcLayerOcl {
    pub fn new(size: usize) -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            ocl_params: None,
            size,
            batch_size: 1,
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
        let kern = self.ocl_kernel.as_mut().unwrap();
        kern.set_arg("prev_shape", sh[0] as i32)
            .expect("[fc_ocl] Failed to set prev_shape arg");

        let queue = self.ocl_queue.as_ref().unwrap();
        // buffer routine
        self.ocl_params =
            Some(init_ocl_params(queue.clone(), self.size, sh).expect("Buffer create failure"));
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        debug!("[fc_ocl] set batch size {}", batch_size);

        self.batch_size = batch_size;

        self.ocl_kernel
            .as_mut()
            .unwrap()
            .set_default_global_work_size(ocl::SpatialDims::One(self.size * self.batch_size));

        self.ocl_kernel
            .as_mut()
            .unwrap()
            .set_arg("batch_size", batch_size as i32)
            .expect("[fc_ocl] Failed to set batch_size arg");

        self.ocl_params = Some(
            fit_to_batch_size_ocl(
                self.ocl_params.as_ref().unwrap().clone(), // TODO : refactor
                self.size,
                batch_size,
                self.ocl_queue.as_ref().unwrap().clone(),
            )
            .expect("Fit to batch size ocl failed"),
        );
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
            .src(FC_LAYER_KERNEL_FWD)
            .build(&ocl_ctx)?;

        let kern = Kernel::builder()
            .name("fc_layer_product")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(self.size * self.batch_size)
            .arg_named("batch_size", 1 as i32)
            .arg_named("prev_shape", 0 as i32)
            .arg_named("self_shape", self.size as i32)
            .arg_named("in", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("out", None::<&Buffer<f32>>)
            .build()?;

        self.ocl_kernel = Some(kern);
        self.ocl_queue = Some(queue);

        Ok(())
    }

    fn forward_ocl(&mut self, params: OclParamsBlob) -> LayerOclResult {
        let prev_params = params.first().unwrap();
        let prev_output = prev_params.output.borrow();
        let self_ws = self.ocl_params.as_ref().unwrap().ws.borrow();
        let self_output = self.ocl_params.as_ref().unwrap().output.borrow();

        let self_kern = self.ocl_kernel.as_mut().unwrap();

        self_kern
            .set_arg("in", &*prev_output)
            .expect("[fc_ocl] Setting param IN failure");
        self_kern
            .set_arg("ws", &*self_ws)
            .expect("[fc_ocl] Setting param WS failure");
        self_kern
            .set_arg("out", &*self_output)
            .expect("[fc_ocl] Setting param OUT failure");

        unsafe {
            self_kern.enq().expect("[fc_ocl] Enqueue failure");
        }

        let mut out_vec = vec![0.0; self.size * self.batch_size];
        let mut in_vec = vec![0.0; prev_output.len()];

        self_output
            .read(&mut out_vec)
            .enq()
            .expect("Failed to read test data");
        prev_output.read(&mut in_vec).enq().unwrap();

        for i in out_vec.iter() {
            print!("{:.2} ", i);
        }
        println!("===");
        for i in in_vec.iter() {
            print!("{:.2} ", i);
        }

        debug!("[fc_ocl] forward");

        Ok(vec![self.ocl_params.as_ref().unwrap().clone()])
    }

    fn backward_ocl(&mut self, prev_input: OclParams, next_input: OclParams) -> LayerOclResult {
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
            batch_size: 1,
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
            batch_size: self.batch_size,
            ocl_kernel: None,
            ocl_queue: Some(queue.clone()),
        }
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}
