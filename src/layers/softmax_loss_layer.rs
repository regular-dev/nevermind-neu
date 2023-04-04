use std::{
    cell::RefCell,
    rc::Rc,
    sync::atomic::{AtomicU32, Ordering},
};

use std::collections::HashMap;
use std::f32::consts::E;

use ndarray::{Array1, Axis, Zip};

use log::{debug, info, warn};

use crate::layers::*;
use crate::learn_params::*;
use crate::util::*;

#[derive(Default, Clone)]
pub struct SoftmaxLossLayer {
    pub size: usize,
    pub lr_params: LearnParams,
}

impl AbstractLayer for SoftmaxLossLayer {
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws_mat = self.lr_params.ws.borrow();
        let ws_mat0 = &ws_mat[0];

        Zip::from(inp_m.rows())
            .and(out_m.rows_mut())
            .par_for_each(|inp_b, out_b| {
                // for each batch
                let mut mul_res = ws_mat0.dot(&inp_b);

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
        expected_vec: Batch,
    ) -> LayerBackwardResult {
        let prev_input = &prev_input[0].output.borrow();
        let mut self_err_vals = self.lr_params.err_vals.borrow_mut();
        let mut self_output = self.lr_params.output.borrow_mut();

        let match_cnt = AtomicU32::new(0);
        let batch_len = self_output.len_of(Axis(0)) as f32;

        Zip::from(self_err_vals.rows_mut())
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

        let accuracy = match_cnt.load(Ordering::SeqCst) as f32 / batch_len;
        self_output[[1, 0]] = accuracy;

        // calc per-weight gradient, TODO : refactor code below
        // for prev_layer :
        let ws = self.lr_params.ws.borrow();
        let mut ws_grad = self.lr_params.ws_grad.borrow_mut();

        for neu_idx in 0..ws[0].shape()[0] {
            for prev_idx in 0..ws[0].shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];

                let mut avg = 0.0;
                Zip::from(prev_input.column(prev_idx))
                    .and(self_err_vals.column(neu_idx))
                    .for_each(|prev_val, err_val| {
                        avg += prev_val * err_val;
                    });

                avg = avg / prev_input.column(prev_idx).len() as f32;

                ws_grad[0][cur_ws_idx] = avg;
            }
        }

        debug!("[ok] SoftmaxLossLayer backward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn layer_type(&self) -> &str {
        "SoftmaxLossLayer"
    }

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.lr_params.clone())
    }

    fn set_learn_params(&mut self, lp: LearnParams) {
        self.lr_params = lp;
    }

    fn size(&self) -> usize {
        self.size
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        let mut copy_l = SoftmaxLossLayer::new(self.size);
        copy_l.set_learn_params(self.lr_params.copy());
        Box::new(copy_l)
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }

    fn set_input_shape(&mut self, sh: &[usize]) {
        let mut lr_params = LearnParams::new(self.size, sh[0]);
        lr_params.output = Rc::new(RefCell::new(Batch::zeros((2, self.size))));
        self.lr_params = lr_params;
    }
}

impl SoftmaxLossLayer {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            lr_params: LearnParams::empty()
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
            self.lr_params = LearnParams::empty();
        }
    }
}
