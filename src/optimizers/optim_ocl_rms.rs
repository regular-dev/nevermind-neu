use crate::layers::*;

use ocl::{Buffer, Context, Device, Kernel, MemFlags, Program, Queue};
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
        
        if (ws_grad[idx] == 0.0) {
            return;
        }

        ws_rms[idx] = alpha * ws_rms[idx] + (1.0 - alpha) * pow(ws_grad[idx], (float)2.0);
        ws[idx] += (learn_rate / (sqrt(ws_rms[idx] + theta))) * ws_grad[idx];
    }
"#;

pub struct OptimizerOclRms {
    pub learn_rate: f32,
    pub alpha: f32,
    pub theta: f32,
    pub ws_rms: HashMap<Uuid, Buffer<f32>>,
    pub queue: Queue,
    pub kernel: Kernel,
}

impl OptimizerOclRms {
    pub fn new(queue: Queue) -> Self {
        let program = Program::builder()
            .devices(queue.device())
            .src(SRC_RMS_KERNEL)
            .build(&queue.context())
            .expect("Failed to create RMS optimizer program");

        let kernel = Kernel::builder()
            .name("rms_optim")
            .program(&program)
            .queue(queue.clone())
            .arg_named("learn_rate", 1e-2 as f32)
            .arg_named("alpha", 0.9 as f32)
            .arg_named("theta", 1e-7 as f32)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("ws_rms", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create RMS optimizer kernel");

        Self {
            learn_rate: 1e-1,
            alpha: 0.9,
            theta: 1e-7,
            ws_rms: HashMap::new(),
            queue,
            kernel,
        }
    }

    pub fn optimize(&mut self, params: OclParams) {
        if !self.ws_rms.contains_key(&params.uuid) {
            let ws_grad = params.ws_grad.borrow();

            let ws_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(ws_grad.len())
                .build()
                .expect("[opt_ocl_sgd] Failed to create ws buffer");

            self.ws_rms.insert(params.uuid, ws_buf);
        }

        let ws_delta_buf = self.ws_rms.get_mut(&params.uuid).unwrap();
        let ws_buf = params.ws.borrow();
        let ws_grad_buf = params.ws_grad.borrow();

        self.kernel
            .set_default_global_work_size(ocl::SpatialDims::One(ws_delta_buf.len()));

        // TODO : learn_rate and momentum will be removed from here
        self.kernel.set_arg("learn_rate", self.learn_rate).unwrap();
        self.kernel.set_arg("alpha", self.alpha).unwrap();
        self.kernel.set_arg("theta", self.theta).unwrap();

        self.kernel
            .set_arg("ws", &*ws_buf)
            .expect("[opt_ocl_sgd] Failed to set WS arg");
        self.kernel
            .set_arg("ws_grad", &*ws_grad_buf)
            .expect("[opt_ocl_sgd] Failed to set WS_GRAD arg");
        self.kernel
            .set_arg("ws_rms", &*ws_delta_buf)
            .expect("[opt_ocl_sgd] Failed to set WS_OPTIM arg");

        unsafe {
            self.kernel.enq().expect("[opt_ocl_sgd] Failed to enqueue kernel");
        }
    }
}
