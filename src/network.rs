use std::collections::HashMap;
use std::vec::Vec;

use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use serde_json;
use serde_yaml;

use log::{debug, error, info, warn};
use uuid::fmt::Simple;

use std::cell::RefCell;
use std::rc::Rc;

use std::error::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
use std::time::Instant;

use ndarray::{Zip, Data};

use ndarray_stats::QuantileExt;

use std::fs::OpenOptions;

use super::dataloader::DataLoader;
use super::learn_params::LearnParams;
use crate::util::Batch;

use super::{
    dataloader::{DataBatch, SimpleDataLoader},
    layers::InputDataLayer,
};

use super::optimizers::*;
use crate::models::Model;

const BENCH_ITER: usize = 500;


/// Neural-Network
pub struct Network<T>
where
    T: Model + Serialize + Clone,
{
    train_dl: Option<Box<dyn DataLoader>>,
    test_dl: Option<Box<dyn DataLoader>>,
    optim: Box<dyn Optimizer>,
    test_err: f32,
    test_batch_size: usize,
    is_write_test_err: bool,
    train_model: Option<T>,
    test_model: Option<T>,
    snap_iter: usize,
    test_iter: usize,
    show_accuracy: bool,
    is_bench_time: bool,
    name: String,
}

impl<T> Network<T>
where
    T: Model + Serialize + Clone,
{
    pub fn new(model: T, just_for_tests: bool) -> Self {
        let mut test_model = None;

        if just_for_tests {
            test_model = Some(model.clone());
        }

        Network {
            train_dl: None,
            test_dl: None,
            optim: Box::new(OptimizerSGD::new(1e-1, 0.8)),
            test_err: 0.0,
            test_batch_size: 10,
            is_write_test_err: true,
            train_model: Some(model),
            test_model,
            snap_iter: 0,
            test_iter: 0,
            is_bench_time: true,
            show_accuracy: true,
            name: "network".to_owned(),
        }
    }

    pub fn test_batch_num(mut self, batch_size: usize) -> Self {
        if let Some(test_model) = self.test_model.as_mut() {
            // TODO : set_batch_size change for prepare_for_tests ?
            test_model.set_batch_size_for_tests(batch_size);
        } else {
            self.test_batch_size = batch_size;
        }

        self
    }

    pub fn test_dataloader(mut self, test_dl: Box<dyn DataLoader>) -> Self {
        self.test_dl = Some(test_dl);
        let s = self.test_batch_num(1);
        s
    }

    pub fn create_test_solver(&mut self) {
        self.test_model = self.train_model.clone();
    }

    pub fn write_test_err_to_file(mut self, state: bool) -> Self {
        self.is_write_test_err = state;
        self
    }

    pub fn save_network_cfg(&mut self, path: &str) -> std::io::Result<()> {
      //  save_solver_cfg(&self.train_model, path)
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

    pub fn is_bench_time(mut self, state: bool) -> Self {
        self.is_bench_time = state;
        self
    }

    pub fn save_model_state(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(train_model) = self.train_model.as_ref() {
            train_model.save_state(path)?
        }
        Ok(()) // TODO : Return error if there is no train_model
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

    /// Test net and returns and average error
    fn test_net(&mut self) -> f32 {
        if self.test_dl.is_none() {
            let err = self.test_err / self.test_batch_size as f32; // error average
            self.test_err = 0.0;
            return err;
        }

        let test_dl = self.test_dl.as_mut().unwrap();
        let mut err = 0.0;

        let test_batch = test_dl.next_batch(self.test_batch_size);
        self.test_model.as_mut().unwrap().feedforward(test_batch.input);

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

    pub fn eval(&mut self, train_data: Batch) -> Rc<RefCell<Batch>> {
        if let Some(test_model) = self.test_model.as_mut() {
            info!("TEST");
            test_model.feedforward(train_data);

            let last_lp = test_model
                .last_layer()
                .learn_params()
                .unwrap();
    
            return last_lp.output.clone();
        }

        if let Some(train_model) = self.train_model.as_mut() {
            train_model.feedforward(train_data);

            let last_lp = train_model.last_layer().learn_params().unwrap();

            return last_lp.output.clone();
        }

        error!("Error evaluation !!!");

        return Rc::new(RefCell::new(Batch::zeros((1,1)))); // TODO : return some error
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

    fn perform_step(&mut self) {
        if self.train_dl.is_none() {
            error!("Train dataset isn't set !!!");
            return;
        }

        if let Some(train_model) = self.train_model.as_mut() {
            let data = self.train_dl.as_ref().unwrap().next_batch(train_model.batch_size());
            train_model.feedforward(data.input);
            train_model.backpropagate(data.output);

            Network::optimize_model(train_model, &mut self.optim);
            
            if self.test_dl.is_none() {
                let lr = train_model.last_layer()
                    .learn_params()
                    .unwrap();
                self.test_err += Self::calc_avg_err(&lr);
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

            if iter_num != 0 && iter_num % BENCH_ITER == 0 && self.is_bench_time {
                let elapsed = bench_time.elapsed();
                info!("Do {} Iteration for {} millisecs", BENCH_ITER, elapsed.as_millis());
                bench_time = Instant::now();
            }

            self.perform_step();

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

            iter_num += 1;
        }

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
