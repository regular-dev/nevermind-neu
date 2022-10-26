use std::ops::{Deref, DerefMut};

use std::collections::HashMap;
use std::f32::consts::E;

use ndarray::{Axis, Zip, Array1};

use log::{debug, info, warn};

use crate::layers::*;
use crate::learn_params::*;
use crate::util::*;

#[derive(Default)]
pub struct SoftmaxLossLayer {
    pub size: usize,
    pub prev_size: usize,
    pub lr_params: LearnParams,

    // test
    pub batch_id: usize,
    pub batch_size: usize,
    pub arr_accuracy: Array1<f32>,
}

impl AbstractLayer for SoftmaxLossLayer {
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        let inp_m = input[0].output.borrow();
        let mut out_m = self.lr_params.output.borrow_mut();
        let ws_mat = self.lr_params.ws.borrow();

        let mul_res = inp_m.deref() * &ws_mat[0];

        let mut e_rows = mul_res.map_axis(Axis(1), |row| row.sum());
        let e_rows_max = array_helpers::max(&e_rows);
        e_rows = e_rows - e_rows_max;
        e_rows = e_rows.mapv_into(|v| {
            E.powf(v)
        });
        let sum_rows = e_rows.sum();

        Zip::from(out_m.deref_mut())
            .and(&e_rows)
            .par_for_each(|out_el, in_e| {
                *out_el = in_e / sum_rows;
            });

        //info!("SoftmaxLossLayer output : {}", out_m.deref());

        debug!("[ok] SoftmaxLossLayer forward()");

        Ok(vec![self.lr_params.clone()])
    }

    fn backward_output(
        &mut self,
        prev_input: ParamsBlob,
        expected_vec: &DataVec,
    ) -> LayerBackwardResult {
        let prev_input = &prev_input[0].output.borrow();
        let mut self_err_vals = self.lr_params.err_vals.borrow_mut();
        let self_output = self.lr_params.output.borrow();

        // TEST
        let mut v_max = -999999.0;
        let mut v_idx = 0;
        let mut expected_idx = -1;

        for (idx, it) in self_output.iter().enumerate() {
            if *it > v_max {
                v_max = *it;
                v_idx = idx;
            }
        }

        for (idx, it) in expected_vec.iter().enumerate() {
            if *it == 1.0 {
                expected_idx = idx as i32;
                break;
            }
        }

        //info!("Predicted : {} | Expected : {}", v_idx, expected_idx);
        //*self.arr_accuracy.get_mut(self.batch_id).unwrap() = (v_idx == expected_idx as usize) as i32 as f32;

        if v_idx == expected_idx as usize {
            *self.arr_accuracy.get_mut(self.batch_id).unwrap() = 1.0;
        }

        self.batch_id += 1;


        if self.batch_id % self.batch_size == 0 && self.batch_id != 0 {
            warn!("Accuracy : {}", self.arr_accuracy.sum() / self.batch_size as f32);
            self.arr_accuracy = Array1::zeros(self.batch_size);
            self.batch_id = 0;
        }

        // TEST
       // let err = self_output.get(expected_idx as usize).unwrap();

        let ce_err = self_output.get(expected_idx as usize).unwrap();

        Zip::from(self_err_vals.deref_mut())
            .and(self_output.deref())
            .and(expected_vec)
            .par_for_each(|err_val, output, expected| {
                if *expected == 1.0 {
                    debug!("expected == 1.0");
                    *err_val = 1.0 - *output;  
                } else {
                    *err_val = (-1.0) * *output;
                }
            });

        //info!("SoftmaxLossLayer output : {}", self_output);
        //info!("SoftmaxLossLayer errors : {}", self_err_vals);


        // calc per-weight gradient, TODO : refactor code below
        // for prev_layer :
        let ws = self.lr_params.ws.borrow();
        let mut ws_grad = self.lr_params.ws_grad.borrow_mut();

        for neu_idx in 0..ws[0].shape()[0] {
            for prev_idx in 0..ws[0].shape()[1] {
                let cur_ws_idx = [neu_idx, prev_idx];
                ws_grad[0][cur_ws_idx] = prev_input[prev_idx] * self_err_vals[neu_idx];
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

    fn layer_cfg(&self) -> HashMap<String, Variant> {
        let mut cfg = HashMap::new();

        cfg.insert("size".to_owned(), Variant::Int(self.size as i32));
        cfg.insert("prev_size".to_owned(), Variant::Int(self.prev_size as i32));

        cfg
    }

    fn set_layer_cfg(&mut self, cfg: &HashMap<String, Variant>) {
        let (mut size, mut prev_size): (usize, usize) = (0, 0);

        if let Variant::Int(var_size) = cfg.get("size").unwrap() {
            size = *var_size as usize;
        }

        if let Variant::Int(var_prev_size) = cfg.get("prev_size").unwrap() {
            prev_size = *var_prev_size as usize;
        }

        debug!("Error size : {}", size);
        debug!("Prev size : {}", prev_size);

        if size > 0 && prev_size > 0 {
            self.size = size;
            self.prev_size = prev_size;
            self.lr_params = LearnParams::new(self.size, self.prev_size);
        }

        // test 
        self.batch_id = 0;
        self.batch_size = 1000;
        self.arr_accuracy = Array1::zeros(self.batch_size);
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl SoftmaxLossLayer {
    pub fn new(size: usize, prev_size: usize) -> Self {
        Self {
            size,
            prev_size,
            lr_params: LearnParams::new(size, prev_size),
            batch_size: 1000,
            arr_accuracy: Array1::zeros(1000),
            batch_id: 0,
            
        }
    }
}
