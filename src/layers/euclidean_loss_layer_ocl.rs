use crate::layers::*;
use crate::learn_params::*;
use crate::ocl::*;
use crate::util::*;

use log::{debug, warn};

use ocl::MemFlags;
use ocl::{Buffer, Context, Device, Kernel, Program, Queue};

use std::collections::HashMap;

static EUCLIDEAN_LOSS_KERNEL_FWD: &'static str = r#"
    __kernel void euclidean_loss(
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
            
        out[idx] = activation(sum);
    }
"#;

static EUCLIDEAN_LOSS_KERNEL_BWD: &'static str = r#"
    __kernel void euclidean_loss_grad(
                __private int const batch_size,
                __private int const prev_shape,
                __private int const self_shape,
                __global const float *self_out,
                __global const float *prev_out,
                __global const float *labels,
                __global float *neu_grad, // counter
                __global float *ws_grad)
    {
        uint const idx = get_global_id(0);

        for (int i = 0; i < batch_size; ++i) {
            __private int inner_idx = i * self_shape + idx;
            neu_grad[inner_idx] = (labels[inner_idx] - self_out[inner_idx]) * deriv(self_out[inner_idx]);
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

pub struct EuclideanLossLayerOcl {
    cpu_params: LearnParams,
    ocl_params: Option<OclParams>,
    size: usize,
    batch_size: usize,
    ocl_queue: Option<Queue>,
    ocl_kernel: Option<Kernel>,
    ocl_kernel_grad: Option<Kernel>,
    ocl_act_func: OclActivationFunc,
}

impl EuclideanLossLayerOcl {
    pub fn new(size: usize) -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            ocl_params: None,
            size,
            batch_size: 1,
            ocl_queue: None,
            ocl_kernel: None,
            ocl_kernel_grad: None,
            ocl_act_func: OclActivationFunc::Raw,
        }
    }

    pub fn new_with_activation(size: usize, act: OclActivationFunc) -> Self {
        let mut out = EuclideanLossLayerOcl::new(size);
        out.set_activation_function(act);
        out
    }

    /// Must be set before init_ocl() was called
    pub fn set_activation_function(&mut self, act: OclActivationFunc) {
        if self.ocl_kernel.is_some() {
            warn!("Setting ocl activation function, while kernel is already built");
        }

        self.ocl_act_func = act;
    }
}

impl AbstractLayer for EuclideanLossLayerOcl {
    fn layer_type(&self) -> &str {
        "EuclideanLossLayerOcl"
    }

    fn size(&self) -> usize {
        self.size
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;

        self.ocl_kernel
            .as_mut()
            .unwrap()
            .set_default_global_work_size(ocl::SpatialDims::One(self.size * self.batch_size));

        self.ocl_kernel
            .as_mut()
            .unwrap()
            .set_arg("batch_size", batch_size as i32)
            .expect("[euc_ocl] Failed to set batch_size arg");
        self.ocl_kernel_grad
            .as_mut()
            .unwrap()
            .set_arg("batch_size", batch_size as i32)
            .expect("[euc_ocl] Failed to set batch_size arg");

        self.ocl_params = Some(
            fit_to_batch_size_ocl(
                self.ocl_params.as_ref().unwrap().clone(), // TODO : refactor
                self.size,
                batch_size,
                self.ocl_queue.as_ref().unwrap().clone(),
            )
            .expect("Fit to batch size ocl failed"),
        );

        self.cpu_params.fit_to_batch_size(batch_size);
    }

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.cpu_params.clone())
    }

    fn set_learn_params(&mut self, lp: LearnParams) {
        self.cpu_params = lp;
    }

    fn set_input_shape(&mut self, sh: &[usize]) {
        let kern = self.ocl_kernel.as_mut().unwrap();
        kern.set_arg("prev_shape", sh[0] as i32)
            .expect("[euc_ocl] Failed to set prev_shape arg");

        let kern_grad = self.ocl_kernel_grad.as_mut().unwrap();
        kern_grad
            .set_arg("prev_shape", sh[0] as i32)
            .expect("[euc_ocl] Failed to set prev_shape arg");

        let queue = self.ocl_queue.as_ref().unwrap();
        // buffer routine
        self.ocl_params =
            Some(init_ocl_params(queue.clone(), self.size, sh).expect("Buffer create failure"));

        self.cpu_params = LearnParams::new(self.size, sh[0]);
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

impl AbstractLayerOcl for EuclideanLossLayerOcl {
    fn init_ocl(
        &mut self,
        ocl_ctx: &Context,
        device: Device,
        queue: Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fwd_act = match self.ocl_act_func {
            OclActivationFunc::Sigmoid => OCL_ACTIVATION_SIGMOID,
            OclActivationFunc::Tanh => OCL_ACTIVATION_TANH,
            OclActivationFunc::ReLU => OCL_ACTIVATION_RELU,
            OclActivationFunc::Raw => OCL_ACTIVATION_RAW,
            OclActivationFunc::LeakyReLU => OCL_ACTIVATION_LEAKY_RELU,
            _ => todo!(),
        };

        let bwd_act = match self.ocl_act_func {
            OclActivationFunc::Sigmoid => OCL_ACTIVATION_SIGMOID_DERIV,
            OclActivationFunc::Tanh => OCL_ACTIVATION_TANH_DERIV,
            OclActivationFunc::ReLU => OCL_ACTIVATION_RELU_DERIV,
            OclActivationFunc::LeakyReLU => OCL_ACTIVATION_LEAKY_RELU_DERIV,
            OclActivationFunc::Raw => OCL_ACTIVATION_RAW_DERIV,
            _ => todo!(),
        };

        let program_fwd = [fwd_act, EUCLIDEAN_LOSS_KERNEL_FWD].join("\n");
        let program_bwd = [bwd_act, EUCLIDEAN_LOSS_KERNEL_BWD].join("\n");

        let program = Program::builder()
            .devices(device)
            .src(program_fwd)
            .build(&ocl_ctx)?;
        let program_grad = Program::builder()
            .devices(device)
            .src(program_bwd)
            .build(&ocl_ctx)?;

        let kern_fwd = Kernel::builder()
            .name("euclidean_loss")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(self.size * self.batch_size)
            .arg_named("batch_size", 0 as i32)
            .arg_named("prev_shape", 0 as i32)
            .arg_named("self_shape", self.size as i32)
            .arg_named("in", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("out", None::<&Buffer<f32>>)
            .build()?;

        let kern_bwd = Kernel::builder()
            .name("euclidean_loss_grad")
            .program(&program_grad)
            .queue(queue.clone())
            .global_work_size(self.size)
            .arg_named("batch_size", self.batch_size as i32)
            .arg_named("prev_shape", 0 as i32)
            .arg_named("self_shape", self.size as i32)
            .arg_named("self_out", None::<&Buffer<f32>>)
            .arg_named("prev_out", None::<&Buffer<f32>>)
            .arg_named("labels", None::<&Buffer<f32>>)
            .arg_named("neu_grad", None::<&Buffer<f32>>)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .build()?;

        self.ocl_queue = Some(queue);
        self.ocl_kernel = Some(kern_fwd);
        self.ocl_kernel_grad = Some(kern_bwd);

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
            .expect("[euc_ocl] Setting param IN failure");
        self_kern
            .set_arg("ws", &*self_ws)
            .expect("[euc_ocl] Setting param WS failure");
        self_kern
            .set_arg("out", &*self_output)
            .expect("[euc_ocl] Setting param OUT failure");

        unsafe {
            self_kern
                .enq()
                .expect("[euc_ocl] Enqueue forward kernel failure");
        }

        debug!("[euc_ocl] forward");

        Ok(vec![self.ocl_params.as_ref().unwrap().clone()])
    }

    fn backward_output_ocl(
        &mut self,
        prev_input: OclParamsBlob,
        expected: Array2D,
    ) -> LayerOclResult {
        let ocl_queue = self.ocl_queue.as_ref().unwrap();

        let lbl_buf = Buffer::builder()
            .queue(ocl_queue.clone())
            .flags(MemFlags::new().read_only())
            .len(expected.len())
            .copy_host_slice(expected.as_slice().unwrap())
            .build()
            .expect("[euc_ocl] Couldn't create label buffer");

        let self_out = self.ocl_params.as_ref().unwrap().output.borrow();
        let self_neu_grad = self.ocl_params.as_ref().unwrap().neu_grad.borrow_mut();
        let self_ws_grad = self.ocl_params.as_ref().unwrap().ws_grad.borrow_mut();
        let prev_out = prev_input.first().unwrap().output.borrow();

        let self_kern = self.ocl_kernel_grad.as_ref().unwrap();

        self_kern
            .set_arg("self_out", &*self_out)
            .expect("[euc_ocl] Setting param SELF_OUT failure");
        self_kern
            .set_arg("prev_out", &*prev_out)
            .expect("[euc_ocl] Setting param PREV_OUT failure");
        self_kern
            .set_arg("labels", &lbl_buf)
            .expect("[euc_ocl] Setting param LABELS failure");
        self_kern
            .set_arg("neu_grad", &*self_neu_grad)
            .expect("[euc_ocl] Setting param NEU_GRAD failure");
        self_kern
            .set_arg("ws_grad", &*self_ws_grad)
            .expect("[euc_ocl] Setting param WS_GRAD failure");

        unsafe {
            self_kern
                .enq()
                .expect("[euc_ocl] Enqueue backward kernel failure");
        }

        Ok(vec![self.ocl_params.as_ref().unwrap().clone()])
    }

    fn ocl_params(&self) -> Option<OclParams> {
        Some(self.ocl_params.as_ref().unwrap().clone())
    }

    fn set_ocl_params(&mut self, params: OclParams) {
        self.ocl_params = Some(params);
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
            ocl_params: None,
            size: 0,
            batch_size: 1,
            ocl_queue: None,
            ocl_kernel: None,
            ocl_kernel_grad: None,
            ocl_act_func: OclActivationFunc::Raw,
        }
    }
}

impl Clone for EuclideanLossLayerOcl {
    fn clone(&self) -> Self {
        let queue = self.ocl_queue.as_ref().unwrap();

        Self {
            cpu_params: self.cpu_params.clone(),
            ocl_params: self.ocl_params.clone(),
            size: self.size,
            batch_size: self.batch_size,
            ocl_kernel: None,
            ocl_kernel_grad: None,
            ocl_act_func: self.ocl_act_func.clone(),
            ocl_queue: Some(queue.clone()),
        }
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}

impl WithParams for EuclideanLossLayerOcl {
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
