use std::error::Error;
use std::fs::File;

use serde::{Serialize, Deserialize};

use crate::dataloader::DataBatch;
use crate::layers_storage::LayersStorage;
use crate::solvers::*;
use crate::err::*;


pub trait Solver {
    fn setup_network(&mut self, layers: LayersStorage);
    fn feedforward(&mut self, train_data: &DataBatch, print_out: bool);
    fn backpropagate(&mut self, train_data: &DataBatch);
    fn optimize_network(&mut self);

    fn layers(&self) -> &LayersStorage;
    fn batch_size(&self) -> usize;

    fn solver_type(&self) -> &str;

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>>;
    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>>;
}

#[derive(Serialize, Deserialize)]
pub struct SolverSerializeHelper
{
    type_solver: String,
}

pub fn solver_type_from_file(file: &str) -> Result<String, Box<dyn Error>> {
    let solver_file = File::open(file)?;
    let solver_helper: SolverSerializeHelper = serde_yaml::from_reader(solver_file)?;

    return Ok(solver_helper.type_solver);
}

pub struct BatchCounter {
    batch_id: usize,
    pub batch_size: usize,
}

impl BatchCounter {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_id: 0,
            batch_size,
        }
    }

    pub fn batch_id(&self) -> usize {
        self.batch_id
    }

    pub fn reset(&mut self) {
        self.batch_id = 0;
    }

    pub fn is_update(&self) -> bool {
        self.batch_id == (self.batch_size - 1)
    }

    pub fn increment(&mut self) {
        if self.batch_id == (self.batch_size - 1) {
            self.batch_id = 0;
            return;
        }

        self.batch_id += 1;
    }
}

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/mind.serial_pb.rs"));
}
