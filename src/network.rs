use std::collections::HashMap;
use std::vec::Vec;

use serde::{Deserialize, Serialize, Serializer};
use serde_json;
use serde_yaml;

use log::{debug, error, info, warn};

use std::cell::RefCell;
use std::rc::Rc;

use std::fs::File;
use std::io::{ErrorKind, Write};
use std::time::Instant;

use ndarray::{Zip, Axis};

use ndarray_stats::QuantileExt;

use std::fs::OpenOptions;

use super::dataloader::DataLoader;
use super::learn_params::LearnParams;
use crate::err::CustomError;
use crate::util::*;

use super::optimizers::*;
use crate::models::Model;

pub enum CallbackReturnAction {
    None,
    Stop,
    StopAndSave,
}

/// Neural-Network
pub struct Network<T>
where
    T: Model + Serialize + Clone,
{
    pub train_dl: Option<Box<dyn DataLoader>>,
    pub test_dl: Option<Box<dyn DataLoader>>,
    optim: Box<dyn Optimizer>,
    test_err_non_dl: f32,
    test_batch_size: usize,
    is_write_test_err: bool,
    train_model: Option<T>,
    test_model: Option<T>,
    snap_iter: usize,
    test_iter: usize,
    cur_iter_err: f32,
    cur_iter_acc: f32,
    learn_rate_decay: f32,
    decay_step: usize,
    show_accuracy: bool,
    pub name: String,
    // callback fn args : (iteration_number, current iteration loss, accuracy)
    callbacks: Vec<Box<dyn FnMut(usize, f32, f32) -> CallbackReturnAction>>,
}

impl<T> Network<T>
where
    T: Model + Serialize + Clone,
{
    pub fn new(model: T) -> Self {
        let test_model = Some(model.clone());

        Network {
            train_dl: None,
            test_dl: None,
            optim: Box::new(OptimizerRMS::new(1e-2, 0.9)),
            test_err_non_dl: 0.0,
            test_batch_size: 10,
            is_write_test_err: true,
            train_model: Some(model),
            test_model,
            snap_iter: 0,
            test_iter: 0,
            cur_iter_acc: -1.0,
            cur_iter_err: 0.0,
            learn_rate_decay: 1.0,
            decay_step: 0,
            show_accuracy: true,
            name: "network".to_owned(),
            callbacks: Vec::new(),
        }
    }

    pub fn new_for_eval(model: T) -> Self {
        let tbs = model.batch_size();
        let test_net = Network {
            train_dl: None,
            test_dl: None,
            optim: Box::new(OptimizerSGD::new(1e-2, 0.8)),
            test_err_non_dl: 0.0,
            test_batch_size: 1,
            is_write_test_err: true,
            train_model: None,
            test_model: Some(model),
            snap_iter: 0,
            test_iter: 0,
            cur_iter_err: 0.0,
            cur_iter_acc: -1.0,
            learn_rate_decay: 1.0,
            decay_step: 0,
            show_accuracy: true,
            name: "network".to_owned(),
            callbacks: Vec::new(),
        };
        test_net.test_batch_num(tbs)
    }

    pub fn test_batch_num(mut self, batch_size: usize) -> Self {
        if let Some(test_model) = self.test_model.as_mut() {
            test_model.set_batch_size_for_tests(batch_size);
        }

        self.test_batch_size = batch_size;

        self
    }

    pub fn test_dataloader(mut self, test_dl: Box<dyn DataLoader>) -> Self {
        self.test_dl = Some(test_dl);
        let s = self.test_batch_num(1);
        s
    }

    pub fn add_callback(&mut self, c: Box<dyn FnMut(usize, f32, f32) -> CallbackReturnAction>) {
        self.callbacks.push(c);
    }

    pub fn create_test_solver(&mut self) {
        self.test_model = self.train_model.clone();
    }

    pub fn write_test_err_to_file(mut self, state: bool) -> Self {
        self.is_write_test_err = state;
        self
    }

    pub fn train_batch_size(&self) -> Option<usize> {
        if let Some(train_model) = &self.train_model {
            return Some(train_model.batch_size());
        } else {
            return None;
        }
    }

    pub fn save_network_cfg(&mut self, path: &str) -> std::io::Result<()> {
        todo!() // TODO : need to save net.cfg with layers_cfg and optimizer_cfg
    }

    pub fn snap_iter(mut self, snap_each_iter: usize) -> Self {
        self.snap_iter = snap_each_iter;
        self
    }

    pub fn test_iter(mut self, err_iter: usize) -> Self {
        self.test_iter = err_iter;
        self
    }

    pub fn save_model_state(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(train_model) = self.train_model.as_ref() {
            return train_model.save_state(path);
        }
        Err(Box::new(CustomError::Other))
    }

    pub fn set_train_dataset(&mut self, data: Box<dyn DataLoader>) {
        self.train_dl = Some(data)
    }

    pub fn set_test_dataset(&mut self, data: Box<dyn DataLoader>) {
        self.test_dl = Some(data)
    }

    pub fn set_optimizer(&mut self, optim: Box<dyn Optimizer>) {
        self.optim = optim
    }

    pub fn set_learn_rate_decay(&mut self, decay: f32) {
        self.learn_rate_decay = decay
    }

    pub fn set_learn_rate_decay_step(&mut self, step: usize) {
        self.decay_step = step;
    }

    /// Test net and returns and average error
    fn test_net(&mut self) -> f32 {
        if self.test_dl.is_none() {
            let err = self.test_err_non_dl / self.test_batch_size as f32; // error average
            self.test_err_non_dl = 0.0;
            return err;
        }

        let test_dl = self.test_dl.as_mut().unwrap();
        let mut err = 0.0;

        let test_batch = test_dl.next_batch(self.test_batch_size);
        self.test_model
            .as_mut()
            .unwrap()
            .feedforward(test_batch.input);

        let len_layers = self.test_model.as_ref().unwrap().layers_count(); // TODO : maybe somehow refactor ?
        let last_layer = self.test_model.as_ref().unwrap().layer(len_layers - 1);

        let lr = last_layer.learn_params().unwrap();
        let out = lr.output.borrow();

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
            info!("Accuracy : {}", accuracy_cnt);
        }

        err / self.test_batch_size as f32
    }

    pub fn eval_one(
        &mut self,
        data: DataVec,
    ) -> Result<Rc<RefCell<Batch>>, Box<dyn std::error::Error>> {
        if self.test_batch_size != 1 {
            error!("Invalid batch size {} for test model", self.test_batch_size);
            return Err(Box::new(CustomError::Other));
        }

        if let Some(test_model) = self.test_model.as_mut() {
            let data_len = data.len();
            let cvt = data.into_shape((1, data_len)).unwrap();
            test_model.feedforward(cvt);

            let last_lp = test_model.last_layer().learn_params().unwrap();

            return Ok(last_lp.output.clone());
        }

        return Err(Box::new(CustomError::Other));
    }

    pub fn eval(
        &mut self,
        train_data: Batch,
    ) -> Result<Rc<RefCell<Batch>>, Box<dyn std::error::Error>> {
        if let Some(test_model) = self.test_model.as_mut() {
            test_model.feedforward(train_data);

            let last_lp = test_model.last_layer().learn_params().unwrap();

            return Ok(last_lp.output.clone());
        }

        if let Some(train_model) = self.train_model.as_mut() {
            train_model.feedforward(train_data);

            let last_lp = train_model.last_layer().learn_params().unwrap();

            return Ok(last_lp.output.clone());
        }

        error!("Error evaluation !!!");

        return Err(Box::new(CustomError::Other));
    }

    fn calc_avg_err(last_layer_lr: &LearnParams) -> f32 {
        let err = last_layer_lr.err_vals.borrow();

        let sq_sum = err.fold(0.0, |mut sq_sum, el| {
            sq_sum += el.powf(2.0);
            return sq_sum;
        });

        let test_err = (sq_sum / err.nrows() as f32).sqrt();
        return test_err;
    }

    fn calc_accuracy(last_layer_lr: &LearnParams) -> Option<f32> {
        let lr_output = last_layer_lr.output.borrow();

        if lr_output.len_of(Axis(0)) > 1 {
            return Some(lr_output[[1, 0]]);
        }

        return None;
    }

    fn perform_learn_rate_decay(&mut self) {
        let opt_params = self.optim.cfg();

        if let Some(lr) = opt_params.get("learn_rate") {
            if let Variant::Float(lr) = lr {
                let decayed_lr = lr * self.learn_rate_decay;

                info!(
                    "Perfoming learning rate decay : from {}, to {}",
                    lr, decayed_lr
                );

                let mut m = HashMap::new();
                m.insert("learn_rate".to_owned(), Variant::Float(decayed_lr));
                self.optim.set_cfg(&m);
            }
        }
    }

    fn perform_step(&mut self) {
        if self.train_dl.is_none() {
            error!("Train dataset isn't set !!!");
            return;
        }

        if let Some(train_model) = self.train_model.as_mut() {
            let data = self
                .train_dl
                .as_ref()
                .unwrap()
                .next_batch(train_model.batch_size());
            train_model.feedforward(data.input);
            train_model.backpropagate(data.output);

            Network::optimize_model(train_model, &mut self.optim);

            // store current iteration loss and accuracy
            let lr = train_model.last_layer().learn_params().unwrap();

            self.cur_iter_err = Self::calc_avg_err(&lr);
            self.cur_iter_acc = Self::calc_accuracy(&lr).unwrap();

            if self.test_dl.is_none() {
                self.test_err_non_dl += self.cur_iter_err;
            }
        }
    }

    fn optimize_model(train_model: &mut T, optimizer: &mut Box<dyn Optimizer>) {
        for i in 0..train_model.layers_count() {
            let l = train_model.layer(i);

            let mut lp = l.learn_params().unwrap();

            optimizer.optimize_network(&mut lp);
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

    fn append_error(&self, f: &mut File, err: f32) -> Result<(), Box<dyn std::error::Error>> {
        write!(f, "{:.6}\n", err)?;
        Ok(())
    }

    pub fn train_for_error(&mut self, err: f32) -> Result<(), Box<dyn std::error::Error>> {
        self.train_for_error_or_iter(err, 0)
    }

    /// Trains till MSE(error) becomes lower than err or
    /// train iteration more then max_iter.
    /// If err is 0, it will ignore the error threshold.
    /// If max_iter is 0, it will ignore max_iter argument.
    pub fn train_for_error_or_iter(
        &mut self,
        err: f32,
        max_iter: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut iter_num = 0;

        let mut err_file = self.create_empty_error_file()?;

        let mut bench_time = Instant::now();
        let mut test_err = 0.0;

        let mut flag_stop = false;
        let mut flag_save = false;

        loop {
            if self.test_dl.is_none() {
                if iter_num % self.test_batch_size == 0 && iter_num != 0 {
                    test_err = self.test_net();

                    if test_err < err {
                        info!("Reached satisfying error value");
                        break;
                    }
                }
            }

            if self.test_dl.is_some() {
                if iter_num % self.test_iter == 0 && iter_num != 0 {
                    test_err = self.test_net();

                    let elapsed = bench_time.elapsed();

                    info!(
                        "Did {} iterations for {} milliseconds",
                        self.test_iter,
                        elapsed.as_millis()
                    );

                    bench_time = Instant::now();

                    if test_err < err {
                        info!("Reached satisfying error value");
                        break;
                    }
                }
            }

            if max_iter != 0 && iter_num >= max_iter {
                info!("Reached max iteration");
                break;
            }

            self.perform_step();

            if iter_num != 0 && self.decay_step != 0 && iter_num % self.decay_step == 0 {
                self.perform_learn_rate_decay();
            }

            if self.snap_iter != 0 && iter_num % self.snap_iter == 0 && iter_num != 0 {
                let filename = format!("{}_{}.state", self.name, iter_num);
                self.save_model_state(&filename)?;
            }

            if self.test_iter != 0
                && iter_num % self.test_iter == 0
                && iter_num != 0
                && self.is_write_test_err
            {
                info!("On iter {} , error is : {}", iter_num, test_err);
                self.append_error(&mut err_file, test_err)?;
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

        if flag_stop {
            return Ok(());
        }

        info!("Training finished !");
        info!("Trained for error : {}", test_err);
        info!("Iterations : {}", iter_num);

        let filename = format!("{}_{}_final.state", self.name, iter_num);
        self.save_model_state(&filename)?;

        Ok(())
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
            eprintln!("Error (serde-yaml) serializing net layers !!!");
            return Err(std::io::Error::new(ErrorKind::Other, x));
        }
    }

    Ok(())
}
