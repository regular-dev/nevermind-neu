use std::collections::HashMap;
use std::vec::Vec;

use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use serde_json;
use serde_yaml;

use log::{debug, error, info, warn};

use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Write};

use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::dataset::DataLoader;
use crate::mind::dummy_layer::DummyLayer;
use crate::mind::error_layer::ErrorLayer;
use crate::mind::hidden_layer::HiddenLayer;
use crate::mind::input_data_layer::InputDataLayer;
use crate::mind::util::{Blob, DataVec, Variant, WsBlob};

use super::{
    dataset::{DataBatch, SimpleDataLoader},
    input_data_layer,
};

use super::solver::Solver;



/// Neural-Network
pub struct Network {
    dataloader: Box<dyn DataLoader>,
    solver: Solver,
}

impl Network {
    pub fn new(dataloader: Box<dyn DataLoader>) -> Self {
        debug!("Created an neural network!");

        Network {
            dataloader,
            solver: Solver::new(),
        }
    }

    /// Setup the network with [0] - input size, [...] - hidden neurons, [N] - output size
    /// TODO : make this function static and make static constructor for Network class
    pub fn setup_simple_network(&mut self, layers: &Vec<usize>) {
        self.solver.setup_simple_network(layers);
    }

    // pub fn load_network_cfg(&mut self, path: &str) -> Result<()> {

    //     Ok(())
    // }

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
