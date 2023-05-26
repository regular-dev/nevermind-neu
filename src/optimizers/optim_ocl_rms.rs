use crate::ocl::*;
use crate::optimizers::*;

use ocl::{Buffer, Kernel, MemFlags, Program, Queue};
use std::collections::HashMap;
use uuid::Uuid;

static SRC_RMS_KERNEL_WS: &'static str = r#"
    __kernel void rms_optim_ws(
                __private float const learn_rate,
                __private float const alpha,
                __private float const theta,
                __global const float *ws_grad,
                __global float *ws,
                __global float *ws_rms)
    {
        uint const idx = get_global_id(0);
        
        if ( ws_grad[idx] == 0.0 ) {
            return;
        }

        if ( isnan(ws_grad[idx]) ) {
            return;
        }

        ws_rms[idx] = alpha * ws_rms[idx] + (1.0 - alpha) * pow(ws_grad[idx], (float)2.0);
        ws[idx] += (learn_rate / (sqrt(ws_rms[idx] + theta))) * ws_grad[idx];
    }
"#;

static SRC_RMS_KERNEL_BIAS: &'static str = r#"
    __kernel void rms_optim_bias(
                __private float const learn_rate,
                __private float const alpha,
                __private float const theta,
                __global const float *bias_grad,
                __global float *bias,
                __global float *bias_rms)
    {
        uint const idx = get_global_id(0);
        
        if ( bias_grad[idx] == 0.0 ) {
            return;
        }

        if ( isnan(bias_grad[idx]) ) {
            return;
        }

        bias_rms[idx] = alpha * bias_rms[idx] + (1.0 - alpha) * pow(bias_grad[idx], (float)2.0);

        float v = (learn_rate / (sqrt(bias_rms[idx] + theta))) * bias_grad[idx];

        if ( !isnan(v) ) {
            bias[idx] += v;
        }
    }
"#;

pub struct OptimizerOclRms {
    learn_rate: f32,
    alpha: f32,
    theta: f32,
    ws_rms: HashMap<Uuid, (Buffer<f32>, Buffer<f32>)>, // ws, bias
    queue: Queue,
    kernel_ws: Kernel,
    kernel_bias: Kernel,
}

impl OptimizerOclRms {
    pub fn new(learn_rate: f32, queue: Queue) -> Self {
        let program_ws = Program::builder()
            .devices(queue.device())
            .src(SRC_RMS_KERNEL_WS)
            .build(&queue.context())
            .expect("Failed to create RMS ws optimizer program");

        let program_bias = Program::builder()
            .devices(queue.device())
            .src(SRC_RMS_KERNEL_BIAS)
            .build(&queue.context())
            .expect("Failed to create RMS bias optimizer program");

        let kernel_ws = Kernel::builder()
            .name("rms_optim_ws")
            .program(&program_ws)
            .queue(queue.clone())
            .arg_named("learn_rate", learn_rate)
            .arg_named("alpha", 0.9 as f32)
            .arg_named("theta", 1e-7 as f32)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("ws_rms", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create RMS ws optimizer kernel");

        let kernel_bias = Kernel::builder()
            .name("rms_optim_bias")
            .program(&program_bias)
            .queue(queue.clone())
            .arg_named("learn_rate", learn_rate)
            .arg_named("alpha", 0.9 as f32)
            .arg_named("theta", 1e-7 as f32)
            .arg_named("bias_grad", None::<&Buffer<f32>>)
            .arg_named("bias", None::<&Buffer<f32>>)
            .arg_named("bias_rms", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create RMS bias optimizer kernel");

        Self {
            learn_rate,
            alpha: 0.9,
            theta: 1e-8,
            ws_rms: HashMap::new(),
            queue,
            kernel_ws,
            kernel_bias,
        }
    }

    pub fn set_learn_rate(&mut self, learn_rate: f32) {
        self.learn_rate = learn_rate;

        self.kernel_ws
            .set_arg("learn_rate", learn_rate as f32)
            .expect("[OCL_RMS] Failed to set learning rate");
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;

        self.kernel_ws
            .set_arg("alpha", self.alpha)
            .expect("[OCL_RMS] Failed to set alpha");
    }

    pub fn set_theta(&mut self, theta: f32) {
        self.theta = theta;

        self.kernel_ws
            .set_arg("theta", self.theta)
            .expect("[OCL_RMS] Failed to set theta");
    }

    pub fn learn_rate(&self) -> f32 {
        self.learn_rate
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn theta(&self) -> f32 {
        self.theta
    }

    fn optimize_ws(&mut self, params: &OclParams) {
        let ws_delta_buf = self.ws_rms.get_mut(&params.uuid).unwrap();
        let ws_buf = params.ws.borrow();
        let ws_grad_buf = params.ws_grad.borrow();

        self.kernel_ws
            .set_default_global_work_size(ocl::SpatialDims::One(ws_delta_buf.0.len()));

        self.kernel_ws
            .set_arg("ws", &*ws_buf)
            .expect("[opt_ocl_rms] Failed to set WS arg");
        self.kernel_ws
            .set_arg("ws_grad", &*ws_grad_buf)
            .expect("[opt_ocl_rms] Failed to set WS_GRAD arg");
        self.kernel_ws
            .set_arg("ws_rms", &ws_delta_buf.0)
            .expect("[opt_ocl_rms] Failed to set WS_RMS arg");

        unsafe {
            self.kernel_ws
                .enq()
                .expect("[opt_ocl_rms] Failed to enqueue kernel");
        }
    }

    fn optimize_bias(&mut self, params: &OclParams) {
        let bias_rms_buf = self.ws_rms.get_mut(&params.uuid).unwrap();
        let bias_buf = params.bias.borrow();
        let bias_grad_buf = params.neu_grad.borrow();

        self.kernel_bias
            .set_default_global_work_size(ocl::SpatialDims::One(bias_rms_buf.1.len()));

        self.kernel_bias
            .set_arg("bias", &*bias_buf)
            .expect("[opt_ocl_rms] Failed to set BIAS arg");
        self.kernel_bias
            .set_arg("bias_grad", &*bias_grad_buf)
            .expect("[opt_ocl_rms] Failed to set BIAS_GRAD arg");
        self.kernel_bias
            .set_arg("bias_rms", &bias_rms_buf.1)
            .expect("[opt_ocl_rms] Failed to set BIAS_RMS arg");

        unsafe {
            self.kernel_bias
                .enq()
                .expect("[opt_ocl_rms] Failed to enqueue kernel");
        }
    }
}

impl OptimizerOcl for OptimizerOclRms {
    fn optimize_ocl_params(&mut self, params: OclParams) {
        if !self.ws_rms.contains_key(&params.uuid) {
            let ws_grad = params.ws_grad.borrow();
            let bias_grad = params.neu_grad.borrow();

            let ws_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(ws_grad.len())
                .build()
                .expect("[opt_ocl_rms] Failed to create ws buffer");

            let bias_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(bias_grad.len())
                .build()
                .expect("[opt_ocl_rms] Failed to create bias buffer");

            self.ws_rms.insert(params.uuid, (ws_buf, bias_buf));
        }

        self.optimize_ws(&params);
        self.optimize_bias(&params);
    }
}

impl WithParams for OptimizerOclRms {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut out = HashMap::new();

        out.insert("type".to_string(), Variant::String("rmsprop".to_string()));
        out.insert("learning_rate".to_string(), Variant::Float(self.learn_rate));
        out.insert("alpha".to_string(), Variant::Float(self.alpha));

        out
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if let Some(lr) = args.get("learning_rate") {
            if let Variant::Float(lr) = lr {
                self.learn_rate = *lr as f32;
            }
        }
        if let Some(alpha) = args.get("alpha") {
            if let Variant::Float(alpha) = alpha {
                self.alpha = *alpha as f32;
            }
        }
    }
}
