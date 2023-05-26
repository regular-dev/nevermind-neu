use crate::ocl::*;
use crate::optimizers::*;

use ocl::{Buffer, Kernel, MemFlags, Program, Queue};
use std::collections::HashMap;
use std::hash::Hash;
use uuid::Uuid;

static SRC_ADAM_KERNEL: &'static str = r#"
    __kernel void adam_optim(
                __private float const learn_rate,
                __private float const theta,
                __private float const b1,
                __private float const b2,
                __global const float *ws_grad,
                __global float *ws,
                __global float *ws_v,
                __global float *ws_m)
    {
        uint const idx = get_global_id(0);

        if ( ws_grad[idx] == 0.0 ) {
            return;
        }

        ws_m[idx] = b1 * ws_m[idx] + (1.0 - b1) * ws_grad[idx];
        ws_v[idx] = b2 * ws_v[idx] + (1.0 - b2) * ws_grad[idx] * ws_grad[idx];
        ws[idx] += learn_rate / sqrt(ws_v[idx] + theta) * ws_m[idx];
    }
"#;

pub struct OptimizerOclAdam {
    learn_rate: f32,
    theta: f32,
    b1: f32,
    b2: f32,
    ws_v: HashMap<Uuid, Buffer<f32>>,
    ws_m: HashMap<Uuid, Buffer<f32>>,

    queue: Queue,
    kernel: Kernel,
}

impl OptimizerOclAdam {
    pub fn new(learn_rate: f32, queue: Queue) -> Self {
        let program = Program::builder()
            .devices(queue.device())
            .src(SRC_ADAM_KERNEL)
            .build(&queue.context())
            .expect("Failed to create Adam optimizer program");

        let kernel = Kernel::builder()
            .name("adam_optim")
            .program(&program)
            .queue(queue.clone())
            .arg_named("learn_rate", learn_rate)
            .arg_named("theta", 1e-9 as f32)
            .arg_named("b1", 0.9 as f32)
            .arg_named("b2", 0.99 as f32)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("ws_v", None::<&Buffer<f32>>)
            .arg_named("ws_m", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create Adam optimizer kernel");

        Self {
            learn_rate,
            b1: 0.9,
            b2: 0.99,
            theta: 1e-7,
            ws_v: HashMap::new(),
            ws_m: HashMap::new(),
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

    pub fn set_theta(&mut self, theta: f32) {
        self.theta = theta;

        self.kernel
            .set_arg("theta", self.theta)
            .expect("[OCL_RMS] Failed to set theta");
    }

    pub fn learn_rate(&self) -> f32 {
        self.learn_rate
    }

    pub fn theta(&self) -> f32 {
        self.theta
    }
}

impl OptimizerOcl for OptimizerOclAdam {
    fn optimize_ocl_params(&mut self, params: OclParams) {
        if !self.ws_v.contains_key(&params.uuid) {
            let ws_grad = params.ws_grad.borrow();

            let ws_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(ws_grad.len())
                .build()
                .expect("[opt_ocl_adam] Failed to create ws buffer");

            let ws_buf_m = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(ws_grad.len())
                .build()
                .expect("[opt_ocl_adam] Failed to create ws buffer");

            self.ws_v.insert(params.uuid, ws_buf);
            self.ws_m.insert(params.uuid, ws_buf_m);
        }

        let ws_v = self.ws_v.get_mut(&params.uuid).unwrap();
        let ws_m = self.ws_m.get_mut(&params.uuid).unwrap();
        let ws_buf = params.ws.borrow();
        let ws_grad_buf = params.ws_grad.borrow();

        self.kernel
            .set_default_global_work_size(ocl::SpatialDims::One(ws_v.len()));

        self.kernel
            .set_arg("ws", &*ws_buf)
            .expect("[opt_ocl_adam] Failed to set WS arg");
        self.kernel
            .set_arg("ws_grad", &*ws_grad_buf)
            .expect("[opt_ocl_adam] Failed to set WS_GRAD arg");
        self.kernel
            .set_arg("ws_v", &*ws_v)
            .expect("[opt_ocl_adam] Failed to set WS_V arg");
        self.kernel
            .set_arg("ws_m", &*ws_m)
            .expect("[opt_ocl_adam] Failed to set WS_M arg");

        unsafe {
            self.kernel
                .enq()
                .expect("[opt_ocl_adam] Failed to enqueue kernel");
        }
    }
}

impl WithParams for OptimizerOclAdam {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut out = HashMap::new();

        out.insert("type".to_string(), Variant::String("adam".to_string()));
        out.insert("learning_rate".to_string(), Variant::Float(self.learn_rate));
        out.insert("b1".to_string(), Variant::Float(self.b1));
        out.insert("b2".to_string(), Variant::Float(self.b2));

        out
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if let Some(lr) = args.get("learning_rate") {
            if let Variant::Float(lr) = lr {
                self.learn_rate = *lr as f32;
            }
        }
        if let Some(b1) = args.get("b1") {
            if let Variant::Float(b1) = b1 {
                self.b1 = *b1 as f32;
            }
        }
        if let Some(b2) = args.get("b2") {
            if let Variant::Float(b2) = b2 {
                self.b2 = *b2 as f32;
            }
        }
    }
}
