use crate::ocl::*;
use crate::optimizers::*;

use ocl::{Buffer, Kernel, MemFlags, Program, Queue};
use std::collections::HashMap;
use uuid::Uuid;

static SRC_RMS_KERNEL: &'static str = r#"
    __kernel void rms_optim(
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

        ws_rms[idx] = alpha * ws_rms[idx] + (1.0 - alpha) * pow(ws_grad[idx], (float)2.0);
        ws[idx] += (learn_rate / (sqrt(ws_rms[idx] + theta))) * ws_grad[idx];
    }
"#;

pub struct OptimizerOclRms {
    learn_rate: f32,
    alpha: f32,
    theta: f32,
    ws_rms: HashMap<Uuid, Buffer<f32>>,
    queue: Queue,
    kernel: Kernel,
}

impl OptimizerOclRms {
    pub fn new(learn_rate: f32, queue: Queue) -> Self {
        let program = Program::builder()
            .devices(queue.device())
            .src(SRC_RMS_KERNEL)
            .build(&queue.context())
            .expect("Failed to create RMS optimizer program");

        let kernel = Kernel::builder()
            .name("rms_optim")
            .program(&program)
            .queue(queue.clone())
            .arg_named("learn_rate", learn_rate)
            .arg_named("alpha", 0.9 as f32)
            .arg_named("theta", 1e-7 as f32)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("ws_rms", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create RMS optimizer kernel");

        Self {
            learn_rate,
            alpha: 0.9,
            theta: 1e-7,
            ws_rms: HashMap::new(),
            queue,
            kernel,
        }
    }

    pub fn set_learn_rate(&mut self, learn_rate: f32) {
        self.learn_rate = learn_rate;

        self.kernel
            .set_arg("learn_rate", learn_rate as f32)
            .expect("[OCL_RMS] Failed to set learning rate");
    }

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;

        self.kernel
            .set_arg("alpha", self.alpha)
            .expect("[OCL_RMS] Failed to set alpha");
    }

    pub fn set_theta(&mut self, theta: f32) {
        self.theta = theta;

        self.kernel
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
}

impl OptimizerOcl for OptimizerOclRms {
    fn optimize_ocl_params(&mut self, params: OclParams) {
        if !self.ws_rms.contains_key(&params.uuid) {
            let ws_grad = params.ws_grad.borrow();

            let ws_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(ws_grad.len())
                .build()
                .expect("[opt_ocl_rms] Failed to create ws buffer");

            self.ws_rms.insert(params.uuid, ws_buf);
        }

        let ws_delta_buf = self.ws_rms.get_mut(&params.uuid).unwrap();
        let ws_buf = params.ws.borrow();
        let ws_grad_buf = params.ws_grad.borrow();

        self.kernel
            .set_default_global_work_size(ocl::SpatialDims::One(ws_delta_buf.len()));

        self.kernel
            .set_arg("ws", &*ws_buf)
            .expect("[opt_ocl_rms] Failed to set WS arg");
        self.kernel
            .set_arg("ws_grad", &*ws_grad_buf)
            .expect("[opt_ocl_rms] Failed to set WS_GRAD arg");
        self.kernel
            .set_arg("ws_rms", &*ws_delta_buf)
            .expect("[opt_ocl_rms] Failed to set WS_OPTIM arg");

        unsafe {
            self.kernel
                .enq()
                .expect("[opt_ocl_rms] Failed to enqueue kernel");
        }
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
