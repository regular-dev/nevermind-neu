use crate::ocl::*;
use crate::optimizers::*;

use ocl::{Buffer, Kernel, MemFlags, Program, Queue};
use std::{collections::HashMap, ops::Deref};

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

        ws_rms[idx] = alpha * ws_rms[idx] + (1.0 - alpha) * pow(ws_grad[idx], (float)2.0);
        ws[idx] += (learn_rate / (sqrt(ws_rms[idx] + theta))) * ws_grad[idx];
    }
"#;

static SRC_RMS_KERNEL_AVG: &'static str = r#"
    __kernel void rms_optim_avg(
                __private float const learn_rate,
                __private float const alpha,
                __private float const theta,
                __private int const batch_size,
                __global const float *grad,
                __global float *buf,
                __global float *rms)
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

        rms[idx] = alpha * rms[idx] + (1.0 - alpha) * pow(avg_grad, (float)2.0);
        buf[idx] += (learn_rate / (sqrt(rms[idx] + theta))) * avg_grad;
    }
"#;

pub struct OptimizerOclRms {
    learn_rate: f32,
    alpha: f32,
    theta: f32,
    rms: HashMap<u64, HashMap<i32, Buffer<Float>>>,

    queue: Queue,
    kernel: Kernel,
    kernel_avg: Kernel,
}

impl OptimizerOclRms {
    pub fn new(learn_rate: f32, queue: Queue) -> Self {
        let program_opt = Program::builder()
            .devices(queue.device())
            .src(SRC_RMS_KERNEL_WS)
            .build(&queue.context())
            .expect("Failed to create RMS ws optimizer program");

        let kernel_opt = Kernel::builder()
            .name("rms_optim_ws")
            .program(&program_opt)
            .queue(queue.clone())
            .arg_named("learn_rate", learn_rate)
            .arg_named("alpha", 0.9 as f32)
            .arg_named("theta", 1e-7 as f32)
            .arg_named("ws_grad", None::<&Buffer<f32>>)
            .arg_named("ws", None::<&Buffer<f32>>)
            .arg_named("ws_rms", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create RMS optimizer kernel");

        let program_opt_avg = Program::builder()
            .devices(queue.device())
            .src(SRC_RMS_KERNEL_AVG)
            .build(&queue.context())
            .expect("Failed to create RMS avg program");

        let kernel_opt_avg = Kernel::builder()
            .name("rms_optim_avg")
            .program(&program_opt_avg)
            .queue(queue.clone())
            .arg_named("learn_rate", learn_rate)
            .arg_named("alpha", 0.9 as f32)
            .arg_named("theta", 1e-7 as f32)
            .arg_named("batch_size", 0 as i32)
            .arg_named("grad", None::<&Buffer<f32>>)
            .arg_named("buf", None::<&Buffer<f32>>)
            .arg_named("rms", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create RMS opt avg optimizer kernel");

        Self {
            learn_rate,
            alpha: 0.9,
            theta: 1e-8,
            rms: HashMap::new(),
            queue,
            kernel: kernel_opt,
            kernel_avg: kernel_opt_avg,
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
    fn optimize_ocl_params(&mut self, params: OclParams, opt_prms: TrainableBufsIds) {
        if !self.rms.contains_key(&params.id) {
            self.rms.insert(params.id, HashMap::new());
        }

        for (buf_id, buf_grad_id) in opt_prms.0.iter().zip(opt_prms.1.iter()) {
            let buf_grad = params.get_buf(*buf_grad_id);
            let buf_grad = buf_grad.0.borrow();

            let rms = self.rms.get_mut(&params.id).unwrap();

            if !rms.contains_key(buf_grad_id) {
                let zeroed_param = OclParams::create_empty_buf(
                    buf_grad.len(),
                    MemFlags::new().read_write(),
                    self.queue.clone(),
                )
                .expect("[ocl_rms] Failed to create V ocl buffer");
                rms.insert(*buf_grad_id, zeroed_param);
            }

            let rms_m = rms.get_mut(buf_grad_id).unwrap();

            let buf = params.get_buf(*buf_id);
            let buf = buf.0.borrow();

            if buf.len() == buf_grad.len() {
                self.kernel
                    .set_default_global_work_size(ocl::SpatialDims::One(buf.len()));

                self.kernel
                    .set_arg("ws", buf.deref())
                    .expect("[opt_ocl_rms] Failed to set WS arg");
                self.kernel
                    .set_arg("ws_grad", buf_grad.deref())
                    .expect("[opt_ocl_rms] Failed to set WS_GRAD arg");
                self.kernel
                    .set_arg("ws_rms", rms_m)
                    .expect("[opt_ocl_rms] Failed to set WS_RMS arg");

                unsafe {
                    self.kernel
                        .enq()
                        .expect("[opt_ocl_rms] Failed to enqueue kernel");
                }
            } else if buf_grad.len() % buf.len() == 0 {
                let batch_size = buf_grad.len() / buf.len();

                self.kernel_avg
                    .set_default_global_work_size(ocl::SpatialDims::One(buf.len()));

                self.kernel_avg
                    .set_arg("batch_size", batch_size as i32)
                    .expect("[opt_ocl_rms] Failed to set batch size");
                self.kernel_avg
                    .set_arg("grad", buf_grad.deref())
                    .expect("[opt_ocl_rms] Faield to set GRAD");
                self.kernel_avg
                    .set_arg("buf", buf.deref())
                    .expect("[opt_ocl_rms] Failed to set BUF");
                self.kernel_avg
                    .set_arg("rms", rms_m)
                    .expect("[opt_ocl_rms] Failed to set WS_RMS arg");

                unsafe {
                    self.kernel_avg
                        .enq()
                        .expect("[opt_ocl_rms] Failed to enqueue avg kernel");
                }
            } else {
                panic!(
                    "[opt_ocl_rms] Invalid buf and grad length : {} | {}",
                    buf.len(),
                    buf_grad.len()
                );
            }
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
