use crate::layers::*;
use crate::cpu_params::*;
use crate::ocl::*;
use crate::util::*;

use log::{debug, warn};

use rand::{thread_rng, Rng, ThreadRng};

use ocl::{Buffer, Context, Device, Kernel, MemFlags, Program, Queue};
use std::{collections::HashMap, error::Error};

static FC_LAYER_KERNEL_FWD: &'static str = r#"
    __kernel void fc_layer_product(
                __private int const batch_size,
                __private int const prev_shape,
                __private int const self_shape,
                __private int const dropout_idx,
                __private int const dropout_len,
                __global const float *in,
                __global const float *bias,
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

        out[idx] = activation(sum + bias[real_idx]);

        if (real_idx >= dropout_idx && real_idx < dropout_idx + dropout_len) {
            out[idx] = 0.0;
        }
    }
"#;

static FC_LAYER_KERNEL_BWD: &'static str = r#"
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

            neu_grad[i * self_shape + idx] = sum_err * deriv(self_out[i * self_shape + idx]);
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
    cpu_params: CpuParams,
    ocl_params: OclParams,
    size: usize,
    batch_size: usize,

    dropout: f32,
    rng: ThreadRng,

    ocl_kernel: Option<Kernel>,
    ocl_kernel_grad: Option<Kernel>,
    ocl_queue: Option<Queue>,
    ocl_act_func: OclActivationFunc,
}

impl FcLayerOcl {
    pub fn new(size: usize, act: OclActivationFunc) -> Self {
        Self {
            cpu_params: CpuParams::empty(),
            ocl_params: OclParams::empty(),
            size,
            batch_size: 1,
            ocl_kernel: None,
            ocl_queue: None,
            ocl_kernel_grad: None,
            ocl_act_func: act,
            dropout: 0.0,
            rng: thread_rng(),
        }
    }

    /// Must be set before init_ocl() was called
    pub fn set_activation_function(&mut self, act: OclActivationFunc) {
        if self.ocl_kernel.is_some() {
            warn!("Setting ocl activation function, while kernel is already built");
        }

        self.ocl_act_func = act;
    }

    pub fn set_dropout(&mut self, dropout: f32) {
        self.dropout = dropout;
    }

    pub fn dropout(&self) -> f32 {
        self.dropout
    }
}

impl AbstractLayer for FcLayerOcl {
    fn layer_type(&self) -> &str {
        "FcLayerOcl"
    }

    fn size(&self) -> usize {
        self.size
    }

    fn cpu_params(&self) -> Option<CpuParams> {
        None
    }

    fn set_cpu_params(&mut self, lp: CpuParams) {}

    fn trainable_bufs(&self) -> TrainableBufsIds {
        (
            &[TypeBuffer::Weights as i32, TypeBuffer::Bias as i32],
            &[TypeBuffer::WeightsGrad as i32, TypeBuffer::NeuGrad as i32],
        )
    }

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
            init_ocl_params(queue.clone(), self.size, sh, true).expect("Buffer create failure");
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

        self.ocl_params
            .fit_to_batch_size_ocl(
                self.size,
                batch_size,
                self.ocl_queue.as_ref().unwrap().clone(),
            )
            .expect("Fit to batch size ocl failed");
    }

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
        let fwd_act = match self.ocl_act_func {
            OclActivationFunc::Sigmoid => OCL_ACTIVATION_SIGMOID,
            OclActivationFunc::Tanh => OCL_ACTIVATION_TANH,
            OclActivationFunc::ReLU => OCL_ACTIVATION_RELU,
            OclActivationFunc::Raw => OCL_ACTIVATION_RAW,
            OclActivationFunc::LeakyReLU => OCL_ACTIVATION_LEAKY_RELU,
        };

        let bwd_act = match self.ocl_act_func {
            OclActivationFunc::Sigmoid => OCL_ACTIVATION_SIGMOID_DERIV,
            OclActivationFunc::Tanh => OCL_ACTIVATION_TANH_DERIV,
            OclActivationFunc::ReLU => OCL_ACTIVATION_RELU_DERIV,
            OclActivationFunc::LeakyReLU => OCL_ACTIVATION_LEAKY_RELU_DERIV,
            OclActivationFunc::Raw => OCL_ACTIVATION_RAW_DERIV,
        };

        let program_fwd = [fwd_act, FC_LAYER_KERNEL_FWD].join("\n");
        let program_bwd = [bwd_act, FC_LAYER_KERNEL_BWD].join("\n");

        let program = Program::builder()
            .devices(device)
            .src(program_fwd)
            .build(&ocl_ctx)?;
        let program_grad = Program::builder()
            .devices(device)
            .src(program_bwd)
            .build(&ocl_ctx)?;

        let kern = Kernel::builder()
            .name("fc_layer_product")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(self.size * self.batch_size)
            .arg_named("batch_size", self.batch_size as i32)
            .arg_named("prev_shape", 0 as i32)
            .arg_named("self_shape", self.size as i32)
            .arg_named("dropout_idx", 0 as i32)
            .arg_named("dropout_len", 0 as i32)
            .arg_named("in", None::<&Buffer<f32>>)
            .arg_named("bias", None::<&Buffer<f32>>)
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
        let prev_output = prev_params.get_buf_t(TypeBuffer::Output);
        let prev_output = prev_output.0.borrow();

        let self_ws = self.ocl_params.get_buf_t(TypeBuffer::Weights);
        let self_ws = self_ws.0.borrow();

        let self_output = self.ocl_params.get_buf_t(TypeBuffer::Output);
        let self_output = self_output.0.borrow();

        let self_bias = self.ocl_params.get_buf_t(TypeBuffer::Bias);
        let self_bias = self_bias.0.borrow();

        let self_kern = self.ocl_kernel.as_ref().unwrap();

        // dropout
        let dropout_len = (self.size as f32 * self.dropout) as i32;
        let dropout_idx = self.rng.gen_range(0, self.size - dropout_len as usize);

        self_kern
            .set_arg("in", &*prev_output)
            .expect("[fc_ocl] Setting param IN failure");
        self_kern
            .set_arg("bias", &*self_bias)
            .expect("[fc_ocl] Failed to set BIAS param");
        self_kern
            .set_arg("ws", &*self_ws)
            .expect("[fc_ocl] Setting param WS failure");
        self_kern
            .set_arg("out", &*self_output)
            .expect("[fc_ocl] Setting param OUT failure");
        self_kern
            .set_arg("dropout_idx", dropout_idx as i32)
            .expect("[fc_ocl] Failed to set DROPOUT_IDX");
        self_kern
            .set_arg("dropout_len", dropout_len as i32)
            .expect("[fc_ocl] Failed to set DROPOUT LEN");

        unsafe {
            self_kern
                .enq()
                .expect("[fc_ocl] Enqueue forward kernel failure");
        }

        debug!("[fc_ocl] forward");

        Ok(vec![self.ocl_params.clone()])
    }

    fn backward_ocl(
        &mut self,
        prev_input: OclParamsBlob,
        next_input: OclParamsBlob,
    ) -> LayerOclResult {
        let self_out = self.ocl_params.get_buf_t(TypeBuffer::Output);
        let self_out = self_out.0.borrow();

        let self_neu_grad = self.ocl_params.get_buf_t(TypeBuffer::NeuGrad);
        let self_neu_grad = self_neu_grad.0.borrow();

        let self_ws_grad = self.ocl_params.get_buf_t(TypeBuffer::WeightsGrad);
        let self_ws_grad = self_ws_grad.0.borrow_mut();

        let prev_out = prev_input.first().unwrap().get_buf_t(TypeBuffer::Output);
        let prev_out = prev_out.0.borrow();

        let next_ws = next_input.first().unwrap().get_buf_t(TypeBuffer::Weights);
        let next_ws = next_ws.0.borrow();

        let next_grad = next_input.first().unwrap().get_buf_t(TypeBuffer::NeuGrad);
        let next_grad = next_grad.0.borrow();

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

        self_kern
            .set_arg("next_shape", next_shape as i32)
            .expect("[fc_ocl] Setting param NEXT_SHAPE failure");

        unsafe {
            self_kern
                .enq()
                .expect("[fc_ocl] Enqueue backward kernel failure");
        }

        debug!("[fc_ocl] backward done");

        Ok(vec![self.ocl_params.clone()])
    }

    fn ocl_params(&self) -> Option<OclParams> {
        Some(self.ocl_params.clone())
    }

    fn set_ocl_params(&mut self, params: OclParams) {
        self.ocl_params = params;
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
            cpu_params: CpuParams::empty(),
            ocl_params: OclParams::empty(),
            size: 0,
            batch_size: 1,
            ocl_kernel: None,
            ocl_kernel_grad: None,
            ocl_queue: None,
            ocl_act_func: OclActivationFunc::Sigmoid,

            dropout: 0.0,
            rng: thread_rng(),
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
            ocl_act_func: self.ocl_act_func.clone(),

            dropout: 0.0,
            rng: thread_rng(),
        }
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}

impl WithParams for FcLayerOcl {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut out = HashMap::new();

        out.insert("size".to_string(), Variant::Int(self.size as i32));
        out.insert(
            "activation".to_string(),
            Variant::String(self.ocl_act_func.to_string()),
        );

        out
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if let Some(size) = args.get("size") {
            if let Variant::Int(size) = size {
                self.size = *size as usize;
            }
        }

        if let Some(act_f) = args.get("activation") {
            if let Variant::String(act_f) = act_f {
                let cvt_res = OclActivationFunc::try_from(act_f.as_str());

                if let Ok(cvt) = cvt_res {
                    self.ocl_act_func = cvt;
                }
            }
        }
    }
}
