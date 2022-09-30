use std::collections::HashMap;
use std::vec::Vec;

use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use serde_json;
use serde_yaml;

use log::{debug, error, info, warn};

use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};
use std::error::*;
use std::time::Instant;

use std::fs::OpenOptions;

use super::dataloader::DataLoader;
use super::layers_storage::LayersStorage;
use super::solvers::SolverRMS;

use super::{
    dataloader::{DataBatch, SimpleDataLoader},
    layers::InputDataLayer,
};

use super::solvers::Solver;


/// Neural-Network
pub struct Network<T> 
where 
    T: Solver + Serialize
{
    dataloader: Box<dyn DataLoader>,
    solver: T,
    snap_iter: usize,
    err_to_file_iter: usize,
    is_bench_time: bool,
    name: String,
}

impl<T> Network<T>
where
    T: Solver + Serialize
{
    pub fn new(dataloader: Box<dyn DataLoader>, solver: T) -> Self {
        Network {
            dataloader,
            solver,
            snap_iter: 0,
            err_to_file_iter: 0,
            is_bench_time: false,
            name: "network".to_owned(),
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    /// TODO : make this function static and make as static constructor for Network class
    pub fn setup_simple_network(&mut self, layers: &Vec<usize>) {
        let ls = LayersStorage::new_simple_network(layers);
        self.solver.setup_network(ls);
    }

    pub fn set_layer_storage(&mut self, ls: LayersStorage) {
        self.solver.setup_network(ls);
    }

    pub fn save_network_cfg(&mut self, path: &str) -> std::io::Result<()> {
        let json_str_result = serde_yaml::to_string(&self.solver);

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

    pub fn snap_iter(mut self, snap_each_iter: usize) -> Self {
        self.snap_iter = snap_each_iter;
        self
    }

    pub fn err_to_file_iter(mut self, err_iter: usize) -> Self {
        self.err_to_file_iter = err_iter;
        self
    }
    
    pub fn is_bench_time(mut self, state: bool) -> Self {
        self.is_bench_time = state;
        self
    }

    pub fn save_solver_state(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.solver.save_state(path)?;
        Ok(())
    }

    pub fn feedforward(&mut self, train_data: &DataBatch, print_out: bool) {
        self.solver.feedforward(train_data, print_out);
    }

    fn perform_step(&mut self) {
        let data = self.dataloader.next();
        self.solver.perform_step(&data);
    }

    pub fn train_for_n_times(&mut self, times: usize) -> Result<(), Box< dyn std::error::Error >> {
        self.train_for_error_or_iter(0.0, times)
    }

    fn create_empty_error_file(&self) -> Result<File, Box<dyn std::error::Error > >{
        let file = OpenOptions::new().write(true)
                             .create(true)
                             .open("err.log")?;
        Ok(file)
    }

    fn append_error(&self, f: &mut File) -> Result<(), Box<dyn std::error::Error>> {
        write!(f, "{:.6}\n", self.solver.error())?;
        Ok(())
    }

    pub fn train_for_error(&mut self, err: f32) -> Result<(), Box<dyn std::error::Error>> {
        self.train_for_error_or_iter(err, 0)
    }

    /// Trains till MSE(error) becomes lower than err or 
    /// train iteration more then max_iter. 
    /// If err is 0, it will ignore the error threshold.
    /// If max_iter is 0, it will ignore max_iter argument.
    pub fn train_for_error_or_iter(&mut self, err: f32, max_iter: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut iter_num = 0;

        let mut err_file = self.create_empty_error_file()?;
        debug!("Squared error : {}", self.solver.error());

        let mut bench_time = Instant::now();

        loop {
            if err != 0.0 && self.solver.error() <= err {
                info!("Reached satisfying error value");
                break;
            }

            if max_iter != 0 && iter_num >= max_iter {
                info!("Reached max iteration");
                break;
            }

            // TODO : make an argument to display info
            if iter_num != 0 && iter_num % 100 == 0 {
                let elapsed = bench_time.elapsed();
                info!("Do {} Iteration for {:.4} secs", 100, elapsed.as_secs_f32());
                bench_time = Instant::now();
            }

            self.perform_step();

            if self.snap_iter != 0 && iter_num % self.snap_iter == 0 && iter_num != 0 {
                let filename = format!("{}_{}.state", self.name, iter_num);
                self.save_solver_state(&filename)?;
            }

            if self.err_to_file_iter != 0 && iter_num % self.err_to_file_iter == 0 && iter_num != 0 {
                info!("on iter {} , error is : {}", iter_num, self.solver.error());
                self.append_error(&mut err_file)?;
            }

            iter_num += 1;
        }

        info!("Trained for error : {}", self.solver.error());
        info!("Iterations : {}", iter_num);

        Ok(())
    }
}
