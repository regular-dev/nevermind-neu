use crate::ocl::*;
use crate::optimizers::*;
use crate::util::*;

use ocl::{Buffer, Kernel, MemFlags, Program, Queue};
use std::{collections::HashMap, ops::Deref};

static SRC_SGD_KERNEL: &'static str = r#"
    __kernel void sgd_optim(
                __private float const learn_rate,
                __private float const momentum,
                __global const float *ws_grad,
                __global float *ws,
                __global float *ws_optim)
    {
        uint const idx = get_global_id(0);
        
        if (ws_grad[idx] == 0.0) {
            return;
        }

        ws[idx] += momentum * ws_optim[idx];
        ws_optim[idx] = learn_rate * ws_grad[idx];
        ws[idx] += ws_optim[idx];
    }
"#;

static SRC_SGD_KERNEL_AVG: &'static str = r#"
    __kernel void sgd_optim_avg(
                __private float const learn_rate,
                __private float const alpha,
                __private float const theta,
                __private int const batch_size,
                __global const float *grad,
                __global float *buf,
                __global float *delta)
    {
        uint const work_size = get_global_size(0);
        uint const idx = get_global_id(0);

        float avg_grad = 0.0;

        for (int i = 0; i < batch_size; ++i) {
            avg_grad += grad[work_size * i + idx];
        }

        avg_grad = avg_grad / batch_size;

        if ( avg_grad == 0.0 ) {
            return;
        }

        buf[idx] += momentum * delta[idx];
        delta[idx] = learn_rate * avg_grad;
        buf[idx] += delta[idx];
    }
"#;

pub struct OptimizerOclSgd {
    pub learn_rate: f32,
    pub momentum: f32,
    pub delta: HashMap<u64, HashMap<i32, Buffer<Float>>>,
    pub queue: Queue,
    pub kernel: Kernel,
    pub kernel_avg: Kernel,
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

        let program_avg = Program::builder()
            .devices(queue.device())
            .src(SRC_SGD_KERNEL_AVG)
            .build(&queue.context())
            .expect("Failed to create SGD avg program");

        let kernel_avg = Kernel::builder()
            .name("sgd_optim_avg")
            .program(&program_avg)
            .queue(queue.clone())
            .arg_named("learn_rate", 1e-2)
            .arg_named("alpha", 0.9 as f32)
            .arg_named("theta", 1e-7 as f32)
            .arg_named("batch_size", 0 as i32)
            .arg_named("grad", None::<&Buffer<f32>>)
            .arg_named("buf", None::<&Buffer<f32>>)
            .arg_named("rms", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create SGD opt avg optimizer kernel");

        Self {
            learn_rate: 1e-1,
            momentum: 0.9,
            delta: HashMap::new(),
            queue,
            kernel,
            kernel_avg,
        }
    }
}

impl OptimizerOcl for OptimizerOclSgd {
    fn optimize_ocl_params(&mut self, params: OclParams, opt_prms: TrainableBufsIds) {
        if !self.delta.contains_key(&params.id) {
            self.delta.insert(params.id, HashMap::new());
        }

        for (buf_id, buf_grad_id) in opt_prms.0.iter().zip(opt_prms.1.iter()) {
            let buf_grad = params.get_buf(*buf_grad_id);
            let buf_grad = buf_grad.0.borrow();

            let delta = self.delta.get_mut(&params.id).unwrap();

            if !delta.contains_key(buf_grad_id) {
                let zeroed_param = OclParams::create_empty_buf(
                    buf_grad.len(),
                    MemFlags::new().read_write(),
                    self.queue.clone(),
                )
                .expect("[ocl_sgd] Failed to create V ocl buffer");
                delta.insert(*buf_grad_id, zeroed_param);
            }

            let delta_m = delta.get_mut(buf_grad_id).unwrap();

            let buf = params.get_buf(*buf_id);
            let buf = buf.0.borrow();

            if buf.len() == buf_grad.len() {
                self.kernel
                    .set_default_global_work_size(ocl::SpatialDims::One(buf.len()));

                self.kernel
                    .set_arg("ws", buf.deref())
                    .expect("[opt_ocl_sgd] Failed to set WS arg");
                self.kernel
                    .set_arg("ws_grad", buf_grad.deref())
                    .expect("[opt_ocl_sgd] Failed to set WS_GRAD arg");
                self.kernel
                    .set_arg("ws_rms", delta_m)
                    .expect("[opt_ocl_sgd] Failed to set WS_RMS arg");

                unsafe {
                    self.kernel
                        .enq()
                        .expect("[opt_ocl_sgd] Failed to enqueue kernel");
                }
            } else if buf_grad.len() % buf.len() == 0 {
                let batch_size = buf_grad.len() / buf.len();

                self.kernel_avg
                    .set_default_global_work_size(ocl::SpatialDims::One(buf.len()));

                self.kernel_avg
                    .set_arg("batch_size", batch_size as i32)
                    .expect("[opt_ocl_sgd] Failed to set batch size");
                self.kernel_avg
                    .set_arg("grad", buf_grad.deref())
                    .expect("[opt_ocl_sgd] Faield to set GRAD");
                self.kernel_avg
                    .set_arg("buf", buf.deref())
                    .expect("[opt_ocl_sgd] Failed to set BUF");
                self.kernel
                    .set_arg("rms", delta_m)
                    .expect("[opt_ocl_sgd] Failed to set WS_RMS arg");

                unsafe {
                    self.kernel_avg
                        .enq()
                        .expect("[opt_ocl_sgd] Failed to enqueue avg kernel");
                }
            } else {
                panic!(
                    "[opt_ocl_sgd] Invalid buf and grad length : {} | {}",
                    buf.len(),
                    buf_grad.len()
                );
            }
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
