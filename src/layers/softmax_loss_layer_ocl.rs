use crate::layers::*;
use crate::learn_params::*;
use crate::ocl::*;
use crate::util::*;

use log::{debug, warn};

use ndarray_stats::QuantileExt;
use ocl::MemFlags;
use ocl::{Buffer, Context, Device, Kernel, Program, Queue};

use ndarray::Zip;

use std::collections::HashMap;

static SOFTMAX_LOSS_KERNEL_FWD: &'static str = r#"
    __kernel void softmax_loss(
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

        __private double e_sum = 0.0;
        __private double act_val = 0.0;
        double max_val = DBL_MIN;

        for (int i = 0; i < self_shape; ++i) {
            __private double sum_i = 0.0;

            for (int j = 0; j < prev_shape; ++j) {
                sum_i += ws[i * prev_shape + j] * in[j + prev_shape * batch_idx];
            }

            if (sum_i > max_val) {
                max_val = sum_i;
            }
        }

        for (int i = 0; i < self_shape; ++i) {
            __private double sum_i = 0.0;

            for (int j = 0; j < prev_shape; ++j) {
                sum_i += ws[i * prev_shape + j] * in[j + prev_shape * batch_idx];
            }

            sum_i -= max_val;

            if (i == real_idx) {
                act_val = exp(sum_i);
            }

            e_sum += exp(sum_i);
        }

        if ( isnan(act_val / e_sum) ) {
            out[idx] = 0.0;
        } else {
            out[idx] = act_val / e_sum;
        }
    }
"#;

static SOFTMAX_LOSS_KERNEL_BWD: &'static str = r#"
    __kernel void softmax_loss_grad(
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
            neu_grad[inner_idx] = (labels[inner_idx] - self_out[inner_idx]);
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

pub struct SoftmaxLossLayerOcl {
    cpu_params: LearnParams,
    ocl_params: Option<OclParams>,
    size: usize,
    batch_size: usize,
    metrics: Metrics,

    ocl_queue: Option<Queue>,
    ocl_kernel: Option<Kernel>,
    ocl_kernel_grad: Option<Kernel>,
}

impl SoftmaxLossLayerOcl {
    pub fn new(size: usize) -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            ocl_params: None,
            size,
            metrics: Metrics::new(),
            batch_size: 1,
            ocl_queue: None,
            ocl_kernel: None,
            ocl_kernel_grad: None,
        }
    }
}

impl AbstractLayer for SoftmaxLossLayerOcl {
    fn layer_type(&self) -> &str {
        "SoftmaxLossLayerOcl"
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

    fn metrics(&self) -> Option<&Metrics> {
        Some(&self.metrics)
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

impl AbstractLayerOcl for SoftmaxLossLayerOcl {
    fn init_ocl(
        &mut self,
        ocl_ctx: &Context,
        device: Device,
        queue: Queue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let program = Program::builder()
            .devices(device)
            .src(SOFTMAX_LOSS_KERNEL_FWD)
            .build(&ocl_ctx)?;
        let program_grad = Program::builder()
            .devices(device)
            .src(SOFTMAX_LOSS_KERNEL_BWD)
            .build(&ocl_ctx)?;

        let kern_fwd = Kernel::builder()
            .name("softmax_loss")
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
            .name("softmax_loss_grad")
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

        // load output to calc accuracy
        // TODO : make calc accuracy optional
        let mut output_vec = WsMat::zeros((self.batch_size, self.size));

        self_out
            .read(output_vec.as_slice_mut().unwrap())
            .enq()
            .expect("Failed to copy OCL buffer to CPU");

        let mut match_cnt = 0;
        Zip::from(output_vec.rows()).and(expected.rows()).for_each(
            |out_arr, lbl_arr| {
                // let (mut arg_max, mut out_max_val) = (-1, 0.0);
                // let mut acc_idx = 0;
                let lbl_idx = lbl_arr.argmax();
                let out_idx = out_arr.argmax();
                
                if lbl_idx == out_idx {
                    match_cnt += 1;
                }
            }
        );

        let accuracy = match_cnt as f64 / self.batch_size as f64;
        self.metrics.insert("accuracy".to_string(), accuracy);

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

impl Default for SoftmaxLossLayerOcl {
    fn default() -> Self {
        Self {
            cpu_params: LearnParams::empty(),
            ocl_params: None,
            size: 0,
            metrics: Metrics::new(),
            batch_size: 1,
            ocl_queue: None,
            ocl_kernel: None,
            ocl_kernel_grad: None,
        }
    }
}

impl Clone for SoftmaxLossLayerOcl {
    fn clone(&self) -> Self {
        let queue = self.ocl_queue.as_ref().unwrap();

        Self {
            cpu_params: self.cpu_params.clone(),
            ocl_params: self.ocl_params.clone(),
            size: self.size,
            batch_size: self.batch_size,
            metrics: self.metrics.clone(),
            ocl_kernel: None,
            ocl_kernel_grad: None,
            ocl_queue: Some(queue.clone()),
        }
    }

    fn clone_from(&mut self, _source: &Self) {
        todo!()
    }
}

impl WithParams for SoftmaxLossLayerOcl {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut out = HashMap::new();
        out.insert("size".to_string(), Variant::Int(self.size as i32));
        out
    }

    fn set_cfg(&mut self, args: &HashMap<String, Variant>) {
        if let Some(size) = args.get("size") {
            if let Variant::Int(size) = size {
                self.size = *size as usize;
            }
        }
    }
}
