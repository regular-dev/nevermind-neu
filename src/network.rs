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

use super::dataset::DataLoader;
use super::layers_storage::LayersStorage;
use super::solvers::SolverRMS;

use super::{
    dataset::{DataBatch, SimpleDataLoader},
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
}

impl<T> Network<T>
where
    T: Solver + Serialize
{
    pub fn new(dataloader: Box<dyn DataLoader>, solver: T) -> Self {
        debug!("Created an neural network!");

        Network {
            dataloader,
            solver,
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    /// TODO : make this function static and make static constructor for Network class
    pub fn setup_simple_network(&mut self, layers: &Vec<usize>) {
        let ls = LayersStorage::new_simple_network(layers);
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

    pub fn train_for_n_times(&mut self, times: i64) {
        for _i in 0..times {
            self.perform_step();
        }
    }
}
