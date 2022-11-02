use std::error::Error;
use std::fs::File;

use serde::{Deserialize, Serialize};

use crate::dataloader::{DataBatch, MiniBatch};
use crate::layers_storage::LayersStorage;
use crate::util::Batch;

pub trait Solver {
    fn setup_network(&mut self, layers: LayersStorage);
    fn feedforward(&mut self, train_data: Batch, print_out: bool); // TODO : remove print_out arg
    fn backpropagate(&mut self, expected_data: Batch);
    fn optimize_network(&mut self);

    fn layers(&self) -> &LayersStorage;
    fn batch_size(&self) -> usize;

    fn solver_type(&self) -> &str;

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>>;
    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>>;
}

#[derive(Serialize, Deserialize)]
pub struct SolverSerializeHelper {
    solver_type: String,
}

pub fn solver_type_from_file(file: &str) -> Result<String, Box<dyn Error>> {
    let solver_file = File::open(file)?;
    let solver_helper: SolverSerializeHelper = serde_yaml::from_reader(solver_file)?;

    return Ok(solver_helper.solver_type);
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
