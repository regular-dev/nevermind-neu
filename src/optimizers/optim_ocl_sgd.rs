use crate::ocl::*;
use crate::util::*;
use crate::optimizers::*;

use ocl::{Buffer, Kernel, MemFlags, Program, Queue};
use std::collections::HashMap;
use uuid::Uuid;

static SRC_SGD_KERNEL: &'static str = r#"
    __kernel void sgd_optim(
                __private float const learn_rate,
                __private float const momentum,
                __global const float *ws_grad,
                __global float *ws,
                __global float *ws_optim)
    {
        uint const idx = get_global_id(0);
        
        ws[idx] += momentum * ws_optim[idx];
        ws_optim[idx] = learn_rate * ws_grad[idx];
        ws[idx] += ws_optim[idx];
    }
"#;

pub struct OptimizerOclSgd {
    pub learn_rate: f32,
    pub momentum: f32,
    pub ws_delta: HashMap<Uuid, Buffer<f32>>,
    pub queue: Queue,
    pub kernel: Kernel,
}

impl OptimizerOclSgd {
    pub fn new(queue: Queue) -> Self {
        let program = Program::builder()
            .devices(queue.device())
            .src(SRC_SGD_KERNEL)
            .build(&queue.context())
            .expect("Failed to create SGD optimizer program");

        let kernel = Kernel::builder()
            .name("sgd_optim")
            .program(&program)
            .queue(queue.clone())
            .arg_named("learn_rate", 1e-2 as f32)
            .arg_named("momentum", 0.9 as f32)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("ws_optim", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create SGD optimizer kernel");

        Self {
            learn_rate: 1e-1,
            momentum: 0.9,
            ws_delta: HashMap::new(),
            queue,
            kernel,
        }
    }
}

impl OptimizerOcl for OptimizerOclSgd {
    fn optimize_ocl_params(&mut self, params: OclParams) {
        if !self.ws_delta.contains_key(&params.uuid) {
            let ws_grad = params.ws_grad.borrow();

            let ws_buf = Buffer::<f32>::builder()
                .queue(self.queue.clone())
                .flags(MemFlags::new().read_write())
                .len(ws_grad.len())
                .build()
                .expect("[opt_ocl_sgd] Failed to create ws buffer");

            self.ws_delta.insert(params.uuid, ws_buf);
        }

        let ws_delta_buf = self.ws_delta.get_mut(&params.uuid).unwrap();
        let ws_buf = params.ws.borrow();
        let ws_grad_buf = params.ws_grad.borrow();

        self.kernel
            .set_default_global_work_size(ocl::SpatialDims::One(ws_delta_buf.len()));

        // TODO : learn_rate and momentum will be removed from here
        self.kernel.set_arg("learn_rate", self.learn_rate).unwrap();
        self.kernel.set_arg("momentum", self.momentum).unwrap();

        self.kernel
            .set_arg("ws", &*ws_buf)
            .expect("[opt_ocl_sgd] Failed to set WS arg");
        self.kernel
            .set_arg("ws_grad", &*ws_grad_buf)
            .expect("[opt_ocl_sgd] Failed to set WS_GRAD arg");
        self.kernel
            .set_arg("ws_optim", &*ws_delta_buf)
            .expect("[opt_ocl_sgd] Failed to set WS_OPTIM arg");

        unsafe {
            self.kernel.enq().expect("[opt_ocl_sgd] Failed to enqueue kernel");
        }
    }
}

impl WithParams for OptimizerOclSgd {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut out = HashMap::new();

        out.insert("type".to_string(), Variant::String("sgd".to_string()));
        out.insert("learning_rate".to_string(), Variant::Float(self.learn_rate));
        out.insert("momentum".to_string(), Variant::Float(self.momentum));

        out
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if let Some(lr) = args.get("learning_rate") {
            if let Variant::Float(lr) = lr {
                self.learn_rate = *lr as f32;
            }
        }
        if let Some(momentum) = args.get("momentum") {
            if let Variant::Float(momentum) = momentum {
                self.momentum = *momentum as f32;
            }
        }
    }
}
