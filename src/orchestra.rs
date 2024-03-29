use std::collections::HashMap;
use std::vec::Vec;

use ndarray_stats::QuantileExt;
use serde::Serialize;
use serde_yaml;

use log::{debug, error, info, warn};

use std::cell::RefCell;
use std::sync::Arc;

use std::fs::File;
use std::io::{ErrorKind, Write};
use std::time::Instant;

use ndarray::{Axis, Zip};

use std::fs::OpenOptions;

use crossbeam::channel;
use std::thread;

use crate::dataloader::*;
use crate::err::CustomError;
use crate::util::*;
use crate::cpu_params::*;

use crate::models::Model;

pub enum CallbackReturnAction {
    None,
    Stop,
    StopAndSave,
}

enum DataloaderMsg {
    Batch(MiniBatch),
    DoNext,
    Pos(usize),
    Stop,
}

/// Neural-Network learning orchestrator
pub struct Orchestra<T>
where
    T: Model + Serialize + Clone,
{
    pub train_dl: Option<Box<dyn DataLoader + Send>>,
    pub test_dl: Option<Box<dyn DataLoader + Send>>,
    test_err_accum: f64,
    test_batch_size: usize,
    is_write_test_err: bool,
    train_model: Option<T>,
    test_model: Option<T>,
    snap_iter: usize,
    test_iter: usize,
    cur_iter_err: f32,
    cur_iter_acc: f64,
    learn_rate_decay: f32,
    decay_step: usize,
    show_accuracy: bool,
    save_on_finish: bool,
    pub name: String,
    // callback fn args : (iteration_number, current iteration loss, accuracy)
    callbacks: Vec<Box<dyn FnMut(usize, f32, f64) -> CallbackReturnAction>>,
}

impl<T> Orchestra<T>
where
    T: Model + Serialize + Clone,
{
    pub fn new(model: T) -> Self {
        let mut test_model = model.clone();
        // Creates separate output buffer for test network,
        // weights buffer is the same as in train model
        test_model.set_batch_size_for_tests(model.batch_size());

        Orchestra {
            train_dl: None,
            test_dl: None,
            test_err_accum: 0.0,
            test_batch_size: 10,
            is_write_test_err: true,
            train_model: Some(model),
            test_model: Some(test_model),
            snap_iter: 0,
            test_iter: 100,
            cur_iter_acc: 0.0,
            cur_iter_err: 0.0,
            learn_rate_decay: 1.0,
            decay_step: 0,
            show_accuracy: true,
            save_on_finish: true,
            name: "network".to_owned(),
            callbacks: Vec::new(),
        }
    }

    pub fn new_for_eval(model: T) -> Self {
        let tbs = model.batch_size();
        let test_net = Orchestra {
            train_dl: None,
            test_dl: None,
            test_err_accum: 0.0,
            test_batch_size: 1,
            is_write_test_err: true,
            train_model: None,
            test_model: Some(model),
            snap_iter: 0,
            test_iter: 100,
            cur_iter_err: 0.0,
            cur_iter_acc: 0.0,
            learn_rate_decay: 1.0,
            decay_step: 0,
            show_accuracy: true,
            save_on_finish: true,
            name: "network".to_owned(),
            callbacks: Vec::new(),
        };
        test_net.test_batch_size(tbs)
    }

    pub fn test_batch_size(mut self, batch_size: usize) -> Self {
        if let Some(test_model) = self.test_model.as_mut() {
            test_model.set_batch_size_for_tests(batch_size);
        }

        self.test_batch_size = batch_size;

        self
    }

    pub fn set_test_batch_size(&mut self, batch_size: usize) {
        self.test_batch_size = batch_size;

        if let Some(test_model) = self.test_model.as_mut() {
            test_model.set_batch_size(self.test_batch_size);
        }
    }

    pub fn test_dataloader(mut self, test_dl: Box<dyn DataLoader + Send>) -> Self {
        self.test_dl = Some(test_dl);
        let s = self.test_batch_size(1);
        s
    }

    pub fn add_callback(&mut self, c: Box<dyn FnMut(usize, f32, f64) -> CallbackReturnAction>) {
        self.callbacks.push(c);
    }

    pub fn create_test_solver(&mut self) {
        self.test_model = self.train_model.clone();
    }

    pub fn write_err_to_file(mut self, state: bool) -> Self {
        self.is_write_test_err = state;
        self
    }

    pub fn set_write_err_to_file(&mut self, state: bool) {
        self.is_write_test_err = state;
    }

    pub fn train_batch_size(&self) -> Option<usize> {
        if let Some(train_model) = &self.train_model {
            return Some(train_model.batch_size());
        } else {
            return None;
        }
    }

    pub fn save_network_cfg(&mut self, _path: &str) -> std::io::Result<()> {
        todo!() // TODO : need to save net.cfg with layers_cfg and optimizer_cfg
    }

    pub fn snap_iter(mut self, snap_each_iter: usize) -> Self {
        self.snap_iter = snap_each_iter;
        self
    }

    pub fn set_snap_iter(&mut self, snap_each_iter: usize) {
        self.snap_iter = snap_each_iter;
    }

    pub fn test_iter(mut self, test_iter: usize) -> Self {
        self.test_iter = test_iter;
        self
    }

    pub fn set_test_iter(&mut self, test_iter: usize) {
        self.test_iter = test_iter;
    }

    pub fn save_model_state(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(train_model) = self.train_model.as_ref() {
            return train_model.save_state(path);
        }
        Err(Box::new(CustomError::Other))
    }

    pub fn set_train_dataset(&mut self, data: Box<dyn DataLoader + Send>) {
        self.train_dl = Some(data)
    }

    pub fn set_test_dataset(&mut self, data: Box<dyn DataLoader + Send>) {
        self.test_dl = Some(data)
    }

    pub fn set_learn_rate_decay(&mut self, decay: f32) {
        self.learn_rate_decay = decay
    }

    pub fn set_learn_rate_decay_step(&mut self, step: usize) {
        self.decay_step = step;
    }

    fn infer_train_error(&mut self, divider_iter: f64) -> f64 {
        let err = self.test_err_accum / divider_iter; // error average
        self.test_err_accum = 0.0;
        return err;
    }

    /// Test net and returns and average error
    fn test_net(&mut self) -> f64 {
        let test_dl = self.test_dl.as_mut().expect("Test dataset isn't set");
        let mut err = 0.0;

        let test_batch = test_dl.next_batch(self.test_batch_size);
        self.test_model
            .as_mut()
            .unwrap()
            .feedforward(test_batch.input);

        let lr = self.test_model.as_ref().unwrap().output_params();
        let out = lr.get_2d_buf_t(TypeBuffer::Output);
        let out = out.borrow();

        let mut accuracy_cnt = 0.0;

        Zip::from(out.rows())
            .and(test_batch.output.rows())
            .for_each(|out_r, exp_r| {
                let mut local_err = 0.0;

                if out_r.argmax() == exp_r.argmax() {
                    accuracy_cnt += 1.0;
                }

                for i in 0..out_r.shape()[0] {
                    local_err += (exp_r[i] - out_r[i]).powf(2.0);
                }

                err += (local_err / out_r.shape()[0] as f32).sqrt();
            });

        accuracy_cnt = accuracy_cnt / self.test_batch_size as f32;

        if self.show_accuracy {
            info!("Validation accuracy : {}", accuracy_cnt);
        }

        err as f64 / self.test_batch_size as f64
    }

    pub fn eval_one(
        &mut self,
        data: DataVec,
    ) -> Result<Arc<RefCell<Array2D>>, Box<dyn std::error::Error>> {
        if self.test_batch_size != 1 {
            error!("Invalid batch size {} for test model", self.test_batch_size);
            return Err(Box::new(CustomError::Other));
        }

        if let Some(test_model) = self.test_model.as_mut() {
            let data_len = data.len();
            let cvt = data.into_shape((1, data_len)).unwrap();
            test_model.feedforward(cvt);

            let last_lp = test_model.output_params();

            return Ok(last_lp.get_2d_buf_t(TypeBuffer::Output));
        }

        return Err(Box::new(CustomError::Other));
    }

    pub fn eval(
        &mut self,
        train_data: Array2D,
    ) -> Result<Arc<RefCell<Array2D>>, Box<dyn std::error::Error>> {
        if let Some(test_model) = self.test_model.as_mut() {
            test_model.feedforward(train_data);

            let last_lp = test_model.output_params();

            return Ok(last_lp.get_2d_buf_t(TypeBuffer::Output));
        }

        if let Some(train_model) = self.train_model.as_mut() {
            train_model.feedforward(train_data);

            let last_lp = train_model.output_params();

            return Ok(last_lp.get_2d_buf_t(TypeBuffer::Output));
        }

        error!("Error evaluation !!!");

        return Err(Box::new(CustomError::Other));
    }

    fn calc_avg_err(last_layer_lr: &CpuParams) -> f32 {
        let err = last_layer_lr.get_2d_buf_t(TypeBuffer::NeuGrad);
        let err = err.borrow();

        let sq_sum = err.fold(0.0, |mut sq_sum, el| {
            sq_sum += el.powf(2.0);
            return sq_sum;
        });

        let test_err = (sq_sum / err.nrows() as f32).sqrt();
        return test_err;
    }

    fn calc_accuracy(metrics: Option<&Metrics>) -> f64 {
        if let Some(metrics) = metrics {
            return metrics["accuracy"];
        } else {
            return 0.0; // none ?
        }
    }

    pub fn set_save_on_finish_flag(&mut self, state: bool) {
        self.save_on_finish = state;
    }

    fn perform_learn_rate_decay(&mut self) {
        let optim = self.train_model.as_mut().unwrap().optimizer_mut();
        let optim_prm = optim.cfg();

        if let Some(lr) = optim_prm.get("learning_rate") {
            if let Variant::Float(lr) = lr {
                let decayed_lr = lr * self.learn_rate_decay;

                info!(
                    "Perfoming learning rate decay : from {}, to {}",
                    lr, decayed_lr
                );

                let mut m = HashMap::new();
                m.insert("learning_rate".to_owned(), Variant::Float(decayed_lr));
                optim.set_cfg(&m);
            }
        } else {
            warn!("Coudln't perform learning rate decay due to cfg entry miss");
        }
    }

    fn perform_step(&mut self, mb: MiniBatch) {
        if let Some(train_model) = self.train_model.as_mut() {
            train_model.feedforward(mb.input);
            train_model.backpropagate(mb.output);

            train_model.optimize();

            // store current iteration loss and accuracy
            let lr = train_model.output_params();

            self.cur_iter_err = Self::calc_avg_err(&lr);
            self.cur_iter_acc = Self::calc_accuracy(train_model.last_layer_metrics());

            self.test_err_accum += self.cur_iter_err as f64;
        }
    }

    pub fn train_for_n_times(&mut self, times: usize) -> Result<(), Box<dyn std::error::Error>> {
        self.train_for_error_or_iter(0.0, times)
    }

    fn create_empty_error_file(&self) -> Result<File, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("err.log")?;
        Ok(file)
    }

    fn append_error(&self, f: &mut File, err: f64) -> Result<(), Box<dyn std::error::Error>> {
        write!(f, "{:.6}\n", err)?;
        Ok(())
    }

    pub fn update_test_model(&mut self) {
        let test_mdl = self.train_model.as_ref().unwrap().clone();
        self.test_model = Some(test_mdl);
    }

    pub fn train_for_error(&mut self, err: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.train_for_error_or_iter(err, 0)
    }

    pub fn train_epochs_or_error(
        &mut self,
        epochs: usize,
        err: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let train_batch_size = self.train_model.as_ref().unwrap().batch_size();
        let iter = epochs
            * self
                .train_dl
                .as_ref()
                .unwrap()
                .len()
                .expect("Train dataset has not length!")
            / train_batch_size;
        self.train_for_error_or_iter(err, iter)
    }

    /// Trains till error becomes lower than err or
    /// train iteration more then max_iter.
    /// If err is 0, it will ignore the error threshold.
    /// If max_iter is 0, it will ignore max_iter argument.
    pub fn train_for_error_or_iter(
        &mut self,
        err: f64,
        max_iter: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut iter_num = 0;

        let mut err_file = None;

        if self.is_write_test_err {
            err_file = Some(self.create_empty_error_file()?);
        }

        let mut bench_time = Instant::now();
        let mut test_err = 0.0;

        let mut flag_stop = false;
        let mut flag_save = false;

        let mut epoch_cnt = 1;
        let ds_len = self.train_dl.as_ref().unwrap().len().unwrap();
        let train_batch_size = self.train_model.as_ref().unwrap().batch_size();

        let ten_perc_metric = ds_len as f64 * 0.1; // for 10% , 20% done displaying
        let mut prev_pos = 0;
        let mut ten_perc_num = 0;

        let mut accuracy_sum = 0.0;

        let (tx_thr, rx_cur) = channel::bounded(2);
        let (tx_cur, rx_thr) = channel::bounded(2);

        let mut train_dl_to_thr = std::mem::replace(&mut self.train_dl, None); // we need to move dataloader to another thread for async batch preparing

        let thread_join = thread::spawn(move || {
            loop {
                let ds_pos = train_dl_to_thr.as_ref().unwrap().pos().unwrap();

                let batch = train_dl_to_thr
                    .as_mut()
                    .unwrap()
                    .next_batch(train_batch_size);

                tx_thr
                    .send(DataloaderMsg::Pos(ds_pos))
                    .expect("Failed to send dataset position from thread");
                tx_thr.send(DataloaderMsg::Batch(batch)).unwrap();

                let resp = rx_thr.recv().unwrap();

                match resp {
                    DataloaderMsg::Stop => break,
                    _ => continue,
                };
            }
            return train_dl_to_thr;
        });

        loop {
            // Error calc for each 10%
            {
                let ds_pos = match rx_cur.recv().unwrap() {
                    DataloaderMsg::Pos(p) => p,
                    _ => panic!("Invalid message"), // TODO : handle without panic
                };

                accuracy_sum += self.cur_iter_acc;

                if train_batch_size * 10 < ds_len // for small datasets do not display percentages
                    && (ds_pos >= (ten_perc_num + 1) as usize * ten_perc_metric as usize
                        || prev_pos > ds_pos)
                {
                    info!(
                        "Done {}% of {} epoch, error : {:.5}",
                        (ten_perc_num + 1) * 10,
                        epoch_cnt,
                        self.test_err_accum
                            / (ten_perc_metric * (ten_perc_num + 1) as f64
                                / train_batch_size as f64),
                    );

                    if accuracy_sum != 0.0 {
                        info!(
                            "Accuracy : {:.4}",
                            accuracy_sum
                                / (ten_perc_metric * (ten_perc_num + 1) as f64
                                    / train_batch_size as f64)
                        );
                    }

                    ten_perc_num += 1;

                    if ten_perc_num > 9 {
                        test_err = self.infer_train_error((ds_len / train_batch_size) as f64); // average error on train dataset

                        if test_err < err {
                            info!("Reached satisfying error value");
                            break;
                        }

                        let elapsed = bench_time.elapsed();

                        info!(
                            "Epoch {} for {:.3} seconds, error : {:.6}",
                            epoch_cnt,
                            elapsed.as_secs_f64(),
                            test_err,
                        );

                        epoch_cnt += 1;
                        ten_perc_num = 0;
                        prev_pos = 0;
                        accuracy_sum = 0.0;

                        bench_time = Instant::now();
                    } else {
                        prev_pos = ds_pos;
                    }
                } else {
                    test_err = self.cur_iter_err as f64;
                }
            }

            // Validation dataset or (testing dataset)
            if self.test_dl.is_some() {
                if iter_num % self.test_iter == 0 && iter_num != 0 {
                    info!("Testing net on {} iteration", iter_num);

                    let val_test_err = self.test_net();

                    info!("Validation error value : {:.5}", val_test_err);

                    if val_test_err < err {
                        info!("Reached satisfying error value on validation dataset!");
                        break;
                    }
                }

                if self.test_iter != 0 && iter_num % self.test_iter == 0 && iter_num != 0 {
                    if self.is_write_test_err {
                        self.append_error(err_file.as_mut().unwrap(), test_err)?;
                    }
                }
            }

            if max_iter != 0 && iter_num >= max_iter {
                info!("Reached max iteration");
                break;
            }

            if let DataloaderMsg::Batch(minibatch) = rx_cur.recv().unwrap() {
                tx_cur.send(DataloaderMsg::DoNext).unwrap();
                self.perform_step(minibatch);
            } else {
                todo!("Handle");
            }

            if iter_num != 0 && self.decay_step != 0 && iter_num % self.decay_step == 0 {
                self.perform_learn_rate_decay();
            }

            if self.snap_iter != 0 && iter_num % self.snap_iter == 0 && iter_num != 0 {
                let filename = format!("{}_{}.state", self.name, iter_num);
                self.save_model_state(&filename)?;
            }

            for it_cb in self.callbacks.iter_mut() {
                let out = it_cb(iter_num, self.cur_iter_err, self.cur_iter_acc);

                match out {
                    CallbackReturnAction::None => (),
                    CallbackReturnAction::Stop => {
                        info!("Stopping training loop on {} iteration...", iter_num);
                        info!("Last test error : {}", test_err);
                        flag_stop = true;
                    }
                    CallbackReturnAction::StopAndSave => {
                        info!("Stopping training loop on {} iteration...", iter_num);
                        info!("Last test error : {}", test_err);
                        flag_save = true;
                        flag_stop = true;
                    }
                }
            }

            if flag_stop {
                break;
            }

            iter_num += 1;
        }

        if flag_save {
            let filename = format!("{}_{}_int.state", self.name, iter_num);
            info!("Saving net to file {}", filename);
            self.save_model_state(&filename)?;
        }

        tx_cur.send(DataloaderMsg::Stop).unwrap();
        let train_dl = thread_join
            .join()
            .expect("Failed to join dataloader thread");
        self.train_dl = train_dl;

        if flag_stop {
            return Ok(());
        }

        info!("Training finished !");
        info!("Trained for error : {}", test_err);
        info!("Iterations : {}", iter_num);

        if self.save_on_finish {
            let filename = format!("{}_{}_final.state", self.name, iter_num);
            self.save_model_state(&filename)?;
        }

        Ok(())
    }

    pub fn train_model(&self) -> Option<&T> {
        self.train_model.as_ref()
    }

    pub fn test_model(&self) -> Option<&T> {
        self.test_model.as_ref()
    }

    pub fn train_model_mut(&mut self) -> Option<&mut T> {
        self.train_model.as_mut()
    }

    pub fn test_model_mut(&mut self) -> Option<&mut T> {
        self.test_model.as_mut()
    }
}

/// TODO : rename this method. it make a confusion with save_network_cfg ?
pub fn save_model_cfg<S: Model + Serialize>(solver: &S, path: &str) -> std::io::Result<()> {
    let yaml_str_result = serde_yaml::to_string(solver);

    let mut output = File::create(path)?;

    match yaml_str_result {
        Ok(yaml_str) => {
            output.write_all(yaml_str.as_bytes())?;
        }
        Err(x) => {
            error!("Error (serde-yaml) serializing net layers !!!");
            return Err(std::io::Error::new(ErrorKind::Other, x));
        }
    }

    Ok(())
}
