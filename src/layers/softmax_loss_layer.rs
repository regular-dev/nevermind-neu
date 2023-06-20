use std::{
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU32, Ordering},
    },
};

use std::collections::HashMap;
use std::f32::consts::E;

use ndarray::{Array1, Axis, Zip, indices};

use log::{debug, info, warn};

use crate::cpu_params::*;
use crate::layers::*;
use crate::util::*;

#[derive(Default, Clone)]
pub struct SoftmaxLossLayer {
    pub size: usize,
    pub lr_params: CpuParams,
    metrics: HashMap<String, f64>,
}

impl AbstractLayer for SoftmaxLossLayer {
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].get_2d_buf_t(TypeBuffer::Output);
        let inp_m = inp_m.borrow();
        let inp_m = inp_m.deref();

        let out_m = self.lr_params.get_2d_buf_t(TypeBuffer::Output);
        let mut out_m = out_m.borrow_mut();
        let out_m = out_m.deref_mut();

        let ws = self.lr_params.get_2d_buf_t(TypeBuffer::Weights);
        let ws = ws.borrow();
        let ws = ws.deref();

        Zip::from(inp_m.rows())
            .and(out_m.rows_mut())
            .par_for_each(|inp_b, out_b| {
                // for each batch
                let mut mul_res = ws.dot(&inp_b);

                // let mut e_rows = mul_res.map_axis(Axis(1), |row| row.sum());
                let e_rows_max = array_helpers::max(&mul_res);
                mul_res = mul_res - e_rows_max;
                mul_res = mul_res.mapv_into(|v| E.powf(v));
                let sum_rows = mul_res.sum();

                Zip::from(out_b).and(&mul_res).for_each(|out_el, in_e| {
                    // for each "neuron"
                    *out_el = in_e / sum_rows;
                });
            });

        debug!("[ok] SoftmaxLossLayer forward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn backward_output(
        &mut self,
        prev_input: ParamsBlob,
        expected_vec: Array2D,
    ) -> LayerBackwardResult {
        let prev_input = &prev_input[0].get_2d_buf_t(TypeBuffer::Output);
        let prev_input = prev_input.borrow();
        let prev_input = prev_input.deref();

        let self_neu_grad = self.lr_params.get_2d_buf_t(TypeBuffer::NeuGrad);
        let mut self_neu_grad = self_neu_grad.borrow_mut();
        let self_neu_grad = self_neu_grad.deref_mut();

        let self_output = self.lr_params.get_2d_buf_t(TypeBuffer::Output);
        let mut self_output = self_output.borrow_mut();
        let self_output = self_output.deref_mut();

        let match_cnt = AtomicU32::new(0);
        let batch_len = self_output.len_of(Axis(0)) as f64;

        Zip::from(self_neu_grad.rows_mut())
            .and(self_output.rows())
            .and(expected_vec.rows())
            .par_for_each(|err_val_b, out_b, expected_b| {
                // for each batch
                let (mut out_idx, mut out_max_val) = (-1, 0.0);
                let mut idx = 0;
                let mut b_expected_idx = -1;

                Zip::from(err_val_b).and(out_b).and(expected_b).for_each(
                    |err_val, output, expected| {
                        if *output > out_max_val {
                            out_max_val = *output;
                            out_idx = idx;
                        }

                        if *expected == 1.0 {
                            b_expected_idx = idx;
                            *err_val = 1.0 - *output;
                        } else {
                            *err_val = (-1.0) * *output;
                        }

                        idx += 1;
                    },
                );

                if b_expected_idx == out_idx {
                    match_cnt.fetch_add(1, Ordering::Relaxed);
                }
            });

        let accuracy = match_cnt.load(Ordering::SeqCst) as f64 / batch_len;
        self.metrics.insert("accuracy".to_string(), accuracy);

        let ws_grad = self.lr_params.get_2d_buf_t(TypeBuffer::WeightsGrad);
        let mut ws_grad = ws_grad.borrow_mut();
        let ws_grad = ws_grad.deref_mut();

        // calc grad for weights
        let ws_idxs = indices(ws_grad.dim());
        Zip::from(ws_grad)
            .and(ws_idxs)
            .for_each(|val_ws_grad, ws_idx| {
                let self_neu_idx = ws_idx.0;
                let prev_neu_idx = ws_idx.1;

                let mut avg = 0.0;

                Zip::from(prev_input.column(prev_neu_idx))
                    .and(self_neu_grad.column(self_neu_idx))
                    .for_each(|prev_val, err_val| {
                        avg += prev_val * err_val;
                    });

                avg = avg / prev_input.column(prev_neu_idx).len() as f32;

                *val_ws_grad = avg;
            });

        debug!("[ok] SoftmaxLossLayer backward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn layer_type(&self) -> &str {
        "SoftmaxLossLayer"
    }

    fn cpu_params(&self) -> Option<CpuParams> {
        Some(self.lr_params.clone())
    }

    fn set_cpu_params(&mut self, lp: CpuParams) {
        self.lr_params = lp;
    }

    fn size(&self) -> usize {
        self.size
    }

    fn metrics(&self) -> Option<&HashMap<String, f64>> {
        Some(&self.metrics)
    }

    fn trainable_bufs(&self) -> TrainableBufsIds {
        (
            &[TypeBuffer::Weights as i32],
            &[TypeBuffer::WeightsGrad as i32],
        )
    }

    fn serializable_bufs(&self) -> &[i32] {
        &[TypeBuffer::Weights as i32]
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = SoftmaxLossLayer::new(self.size);
        copy_l.set_cpu_params(self.lr_params.copy());
        Box::new(copy_l)
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }

    fn set_input_shape(&mut self, sh: &[usize]) {
        self.lr_params = CpuParams::new(self.size, sh[0]);
    }
}

impl SoftmaxLossLayer {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            lr_params: CpuParams::empty(),
            metrics: HashMap::new(),
        }
    }

    pub fn new_box(size: usize) -> Box<Self> {
        Box::new(SoftmaxLossLayer::new(size))
    }
}

impl WithParams for SoftmaxLossLayer {
    fn cfg(&self) -> HashMap<String, Variant> {
        let mut cfg = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.size as i32));

        cfg
    }

    fn set_cfg(&mut self, cfg: &HashMap<String, Variant>) {
        let mut size = 0;

        if let Some(var_size) = cfg.get("size") {
            if let Variant::Int(var_size) = var_size {
                size = *var_size as usize;
            }
        }

        if size > 0 {
            self.size = size;
            self.lr_params = CpuParams::empty();
        }
    }
}
