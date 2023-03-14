use crate::layers::*;
use crate::learn_params::LearnParams;
use crate::util::*;

use log::debug;

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

        for (int j = 0; j < prev_shape; ++j) {
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

        for (int i = 0; i < batch_size; ++i) {
            __private float sum_err = 0.0;

            for (int j = 0; j < next_shape; ++j) {
                sum_err += next_grad[i * next_shape + j] * next_ws[self_shape * j + idx];
            }

            neu_grad[i * self_shape + idx] = sum_err * sigmoid_deriv(self_out[i * self_shape + idx]);
        }

        for (int i = 0; i < prev_shape; ++i) {
            __private float avg_grad = 0.0;

            for (int j = 0; j < batch_size; ++j) {
                avg_grad += neu_grad[j * self_shape + idx] * prev_out[j * prev_shape + i];
            }

            avg_grad = avg_grad / batch_size;

            ws_grad[idx * prev_shape + i] = avg_grad;
        }
    }
"#;

pub struct FcLayerOcl {
    cpu_params: LearnParams,
    ocl_params: Option<OclParams>,
    size: usize,
    batch_size: usize,

    ocl_kernel: Option<Kernel>,
    ocl_kernel_grad: Option<Kernel>,
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
            ocl_kernel_grad: None,
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
        let kern_grad = self.ocl_kernel_grad.as_mut().unwrap();
        kern.set_arg("prev_shape", sh[0] as i32)
            .expect("[fc_ocl] Failed to set prev_shape arg");
        kern_grad
            .set_arg("prev_shape", sh[0] as i32)
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
        self.ocl_kernel_grad
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
        let program_grad = Program::builder()
            .devices(device)
            .src(FC_LAYER_KERNEL_BWD)
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

        let kern_grad = Kernel::builder()
            .name("fc_layer_grad")
            .program(&program_grad)
            .queue(queue.clone())
            .global_work_size(self.size)
            .arg_named("batch_size", self.batch_size as i32)
            .arg_named("prev_shape", 0 as i32)
            .arg_named("next_shape", 0 as i32)
            .arg_named("self_shape", self.size as i32)
            .arg_named("self_out", None::<&Buffer<f32>>)
            .arg_named("next_grad", None::<&Buffer<f32>>)
            .arg_named("next_ws", None::<&Buffer<f32>>)
            .arg_named("prev_out", None::<&Buffer<f32>>)
            .arg_named("neu_grad", None::<&Buffer<f32>>)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .build()?;

        self.ocl_kernel = Some(kern);
        self.ocl_kernel_grad = Some(kern_grad);
        self.ocl_queue = Some(queue);

        Ok(())
    }

    fn forward_ocl(&mut self, params: OclParamsBlob) -> LayerOclResult {
        let prev_params = params.first().unwrap();
        let prev_output = prev_params.output.borrow();
        let self_ws = self.ocl_params.as_ref().unwrap().ws.borrow();
        let self_output = self.ocl_params.as_ref().unwrap().output.borrow();

        let self_kern = self.ocl_kernel.as_ref().unwrap();

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
            self_kern
                .enq()
                .expect("[fc_ocl] Enqueue forward kernel failure");
        }

        let mut out_vec = vec![0.0; self.size * self.batch_size];
        let mut in_vec = vec![0.0; prev_output.len()];

        self_output
            .read(&mut out_vec)
            .enq()
            .expect("Failed to read test data");
        prev_output.read(&mut in_vec).enq().unwrap();

        debug!("[fc_ocl] forward");

        Ok(vec![self.ocl_params.as_ref().unwrap().clone()])
    }

    fn backward_ocl(
        &mut self,
        prev_input: OclParamsBlob,
        next_input: OclParamsBlob,
    ) -> LayerOclResult {
        let self_out = self.ocl_params.as_ref().unwrap().output.borrow();
        let self_neu_grad = self.ocl_params.as_ref().unwrap().neu_grad.borrow_mut();
        let self_ws_grad = self.ocl_params.as_ref().unwrap().ws_grad.borrow_mut();
        let prev_out = prev_input.first().unwrap().output.borrow();
        let next_ws = next_input.first().unwrap().ws.borrow();
        let next_grad = next_input.first().unwrap().neu_grad.borrow();
        let next_shape = next_grad.len() / self.batch_size;

        let self_kern = self.ocl_kernel_grad.as_ref().unwrap();

        self_kern
            .set_arg("self_out", &*self_out)
            .expect("[fc_ocl] Setting param SELF_OUT failure");
        self_kern
            .set_arg("neu_grad", &*self_neu_grad)
            .expect("[fc_ocl] Setting param NEU_GRAD failure");
        self_kern
            .set_arg("ws_grad", &*self_ws_grad)
            .expect("[fc_ocl] Setting param WS_GRAD failure");
        self_kern
            .set_arg("prev_out", &*prev_out)
            .expect("[fc_ocl] Setting param PREV_OUT failure");
        self_kern
            .set_arg("next_ws", &*next_ws)
            .expect("[fc_ocl] Setting param NEXT_WS failure");
        self_kern
            .set_arg("next_grad", &*next_grad)
            .expect("[fc_ocl] Setting param NEXT_GRAD failure");

        // also set next_shape param
        self_kern
            .set_arg("next_shape", next_shape as i32)
            .expect("[fc_ocl] Setting param NEXT_SHAPE failure");

        unsafe {
            self_kern
                .enq()
                .expect("[fc_ocl] Enqueue backward kernel failure");
        }

        debug!("[fc_ocl] backward done");

        Ok(vec![self.ocl_params.as_ref().unwrap().clone()])
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
            ocl_kernel_grad: None,
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
            ocl_kernel_grad: None,
            ocl_queue: Some(queue.clone()),
        }
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}
