use crate::ocl::*;
use crate::optimizers::*;

use ocl::{Buffer, Kernel, MemFlags, Program, Queue};
use std::{collections::HashMap, ops::Deref};

static SRC_ADAM_KERNEL_WS: &'static str = r#"
    __kernel void adam_optim_ws(
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

static SRC_ADAM_KERNEL_AVG: &'static str = r#"
    __kernel void adam_optim_avg(
                __private float const learn_rate,
                __private float const theta,
                __private float const b1,
                __private float const b2,
                __private int const batch_size,
                __global const float *grad,
                __global float *buf,
                __global float *v,
                __global float *m)
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

        m[idx] = b1 * m[idx] + (1.0 - b1) * avg_grad;
        v[idx] = b2 * v[idx] + (1.0 - b2) * avg_grad * avg_grad;
        buf[idx] += learn_rate / sqrt(v[idx] + theta) * m[idx];
    }
"#;

pub struct OptimizerOclAdam {
    learn_rate: f32,
    theta: f32,
    b1: f32,
    b2: f32,
    v: HashMap<u64, HashMap<i32, Buffer<Float>>>,
    m: HashMap<u64, HashMap<i32, Buffer<Float>>>,

    queue: Queue,
    kernel: Kernel,
    kernel_avg: Kernel,
}

impl OptimizerOclAdam {
    pub fn new(learn_rate: f32, queue: Queue) -> Self {
        let program_opt = Program::builder()
            .devices(queue.device())
            .src(SRC_ADAM_KERNEL_WS)
            .build(&queue.context())
            .expect("Failed to create Adam optimizer program");

        let kernel_opt = Kernel::builder()
            .name("adam_optim_ws")
            .program(&program_opt)
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

        let program_opt_avg = Program::builder()
            .devices(queue.device())
            .src(SRC_ADAM_KERNEL_AVG)
            .build(&queue.context())
            .expect("Failed to create Adam optimizer avg program");

        let kernel_opt_avg = Kernel::builder()
            .name("adam_optim_avg")
            .program(&program_opt_avg)
            .queue(queue.clone())
            .arg_named("learn_rate", learn_rate)
            .arg_named("theta", 1e-9 as f32)
            .arg_named("b1", 0.9 as f32)
            .arg_named("b2", 0.99 as f32)
            .arg_named("batch_size", 0 as i32)
            .arg_named("grad", None::<&Buffer<f32>>)
            .arg_named("buf", None::<&Buffer<f32>>)
            .arg_named("v", None::<&Buffer<f32>>)
            .arg_named("m", None::<&Buffer<f32>>)
            .build()
            .expect("Failed to create Adam optimizer avg kernel");

        Self {
            learn_rate,
            b1: 0.9,
            b2: 0.99,
            theta: 1e-7,
            v: HashMap::new(),
            m: HashMap::new(),
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
    fn optimize_ocl_params(&mut self, params: OclParams, opt_prms: TrainableBufsIds) {
        if !self.v.contains_key(&params.id) {
            self.v.insert(params.id, HashMap::new());
            self.m.insert(params.id, HashMap::new());
        }

        for (buf_id, buf_grad_id) in opt_prms.0.iter().zip(opt_prms.1.iter()) {
            let buf_grad = params.get_buf(*buf_grad_id);
            let buf_grad = buf_grad.0.borrow();

            let v = self.v.get_mut(&params.id).unwrap();
            let m = self.m.get_mut(&params.id).unwrap();

            if !v.contains_key(buf_grad_id) {
                let zeroed_param = OclParams::create_empty_buf(
                    buf_grad.len(),
                    MemFlags::new().read_write(),
                    self.queue.clone(),
                )
                .expect("[ocl_adam] Failed to create V ocl buffer");
                v.insert(*buf_grad_id, zeroed_param);
            }

            if !m.contains_key(buf_grad_id) {
                let zeroed_param = OclParams::create_empty_buf(
                    buf_grad.len(),
                    MemFlags::new().read_write(),
                    self.queue.clone(),
                )
                .expect("[ocl_adam] Failed to create M buffer");
                m.insert(*buf_grad_id, zeroed_param);
            }

            let v_m = v.get_mut(buf_grad_id).unwrap();
            let m_m = m.get_mut(buf_grad_id).unwrap();

            let buf = params.get_buf(*buf_id);
            let buf = buf.0.borrow();

            if buf.len() == buf_grad.len() {
                self.kernel
                    .set_default_global_work_size(ocl::SpatialDims::One(buf.len()));

                self.kernel
                    .set_arg("ws", buf.deref())
                    .expect("[opt_ocl_adam] Failed to set WS arg");
                self.kernel
                    .set_arg("ws_grad", buf_grad.deref())
                    .expect("[opt_ocl_adam] Failed to set WS_GRAD arg");
                self.kernel
                    .set_arg("ws_v", v_m)
                    .expect("[opt_ocl_adam] Failed to set WS_V arg");
                self.kernel
                    .set_arg("ws_m", m_m)
                    .expect("[opt_ocl_adam] Failed to set WS_M arg");

                unsafe {
                    self.kernel
                        .enq()
                        .expect("[opt_ocl_adam] Failed to enqueue opt kernel");
                }
            } else if buf_grad.len() % buf.len() == 0 {
                let batch_size = buf_grad.len() / buf.len();

                self.kernel_avg
                    .set_default_global_work_size(ocl::SpatialDims::One(buf.len()));

                self.kernel_avg
                    .set_arg("batch_size", batch_size as i32)
                    .expect("[opt_ocl_adam] Failed to set batch size");
                self.kernel_avg
                    .set_arg("grad", buf_grad.deref())
                    .expect("[opt_ocl_adam] Faield to set GRAD");
                self.kernel_avg
                    .set_arg("buf", buf.deref())
                    .expect("[opt_ocl_adam] Failed to set BUF");
                self.kernel_avg
                    .set_arg("v", v_m)
                    .expect("[opt_ocl_adam] Failed to set WS_V arg");
                self.kernel_avg
                    .set_arg("m", m_m)
                    .expect("[opt_ocl_adam] Failed to set WS_M arg");

                unsafe {
                    self.kernel_avg
                        .enq()
                        .expect("[opt_ocl_adam] Failed to enqueue avg kernel");
                }
            } else {
                panic!(
                    "[opt_ocl_adam] Invalid buf and grad length : {} | {}",
                    buf.len(),
                    buf_grad.len()
                );
            }
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
