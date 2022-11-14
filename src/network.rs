use std::collections::HashMap;
use std::vec::Vec;

use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use serde_json;
use serde_yaml;

use log::{debug, error, info, warn};

use std::cell::RefCell;
use std::rc::Rc;

use std::error::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
use std::time::Instant;

use ndarray::Zip;

use ndarray_stats::QuantileExt;

use std::fs::OpenOptions;

use super::dataloader::DataLoader;
use super::layers_storage::LayersStorage;
use super::learn_params::LearnParams;
use super::solvers::SolverRMS;
use crate::util::Batch;

use super::{
    dataloader::{DataBatch, SimpleDataLoader},
    layers::InputDataLayer,
};

use super::solvers::Solver;

const BENCH_ITER: usize = 500;

/// Neural-Network
pub struct Network<T>
where
    T: Solver + Serialize + Clone,
{
    train_dl: Box<dyn DataLoader>,
    test_dl: Option<Box<dyn DataLoader>>,
    test_batch_size: usize,
    test_err: f32,
    is_write_test_err: bool,
    train_solver: T,
    test_solver: T,
    snap_iter: usize,
    test_iter: usize,
    show_accuracy: bool,
    is_bench_time: bool,
    name: String,
}

impl<T> Network<T>
where
    T: Solver + Serialize + Clone,
{
    pub fn new(train_dl: Box<dyn DataLoader>, solver: T) -> Self {
        let test_solver = solver.clone();

        Network {
            train_dl,
            test_dl: None,
            test_batch_size: 1,
            test_err: 0.0,
            is_write_test_err: true,
            train_solver: solver,
            test_solver,
            snap_iter: 0,
            test_iter: 0,
            is_bench_time: true,
            show_accuracy: true,
            name: "network".to_owned(),
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    pub fn setup_simple_network(&mut self, layers: &Vec<usize>) {
        let mut ls = LayersStorage::new_simple_network(layers);
        ls.fit_to_batch_size(self.train_solver.batch_size());
        self.train_solver.setup_network(ls);
        self.create_test_solver();
    }

    pub fn set_layer_storage(&mut self, ls: LayersStorage) {
        self.train_solver.setup_network(ls);
    }

    pub fn test_batch_num(mut self, batch_size: usize) -> Self {
        self.test_batch_size = batch_size;

        self.test_solver.set_batch_size(batch_size); // TODO : set_batch_size change for prepare_for_tests ?

        self
    }

    pub fn test_dataloader(mut self, test_dl: Box<dyn DataLoader>) -> Self {
        self.test_dl = Some(test_dl);

        self.prepare_test_solver();

        self
    }

    fn prepare_test_solver(&mut self) {
        self.test_solver
            .layers_mut()
            .prepare_for_tests(self.test_batch_size);
    }

    pub fn create_test_solver(&mut self) {
        self.test_solver = self.train_solver.clone();
    }

    pub fn write_test_err_to_file(mut self, state: bool) -> Self {
        self.is_write_test_err = state;
        self
    }

    pub fn save_network_cfg(&mut self, path: &str) -> std::io::Result<()> {
        save_solver_cfg(&self.train_solver, path)
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

    pub fn save_solver_state(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.train_solver.save_state(path)?;
        Ok(())
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
        self.test_solver.feedforward(test_batch.input);

        let layers = self.test_solver.layers();
        let last_layer = layers.last().unwrap();

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
                    local_err += exp_r[i] - out_r[i];
                }

                err += local_err.powf(2.0) / out_r.shape()[0] as f32;
            });

        accuracy_cnt = accuracy_cnt / self.test_batch_size as f32;

        if self.show_accuracy {
            info!("Accuracy : {}", accuracy_cnt);
        }

        err / self.test_batch_size as f32
    }

    pub fn eval(&mut self, train_data: Batch) -> Rc<RefCell<Batch>> {
        self.test_solver.feedforward(train_data);

        let last_lp = self
            .test_solver
            .layers()
            .last()
            .unwrap()
            .learn_params()
            .unwrap();

        last_lp.output.clone()
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
        let data = self.train_dl.next_batch(self.train_solver.batch_size());
        self.train_solver.feedforward(data.input);
        self.train_solver.backpropagate(data.output);
        self.train_solver.optimize_network();

        if self.test_dl.is_none() {
            let lr = self
                .train_solver
                .layers()
                .last()
                .unwrap()
                .learn_params()
                .unwrap();
            self.test_err += Self::calc_avg_err(&lr);
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
                self.save_solver_state(&filename)?;
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
        self.save_solver_state(&filename)?;

        Ok(())
    }
}

/// TODO : rename this method. it make a confusion with save_network_cfg ?
pub fn save_solver_cfg<S: Solver + Serialize>(solver: &S, path: &str) -> std::io::Result<()> {
    let json_str_result = serde_yaml::to_string(solver);

    let mut output = File::create(path)?;

    match json_str_result {
        Ok(json_str) => {
            output.write_all(json_str.as_bytes())?;
        }
        Err(x) => {
            eprintln!("Error (serde-yaml) serializing net layers !!!");
            return Err(std::io::Error::new(ErrorKind::Other, x));
        }
    }

    Ok(())
}
