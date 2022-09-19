use std::error::Error;

use crate::dataloader::DataBatch;
use crate::layers_storage::LayersStorage;


pub trait Solver {
    fn setup_network(&mut self, layers: LayersStorage);
    fn perform_step(&mut self, data: &DataBatch);
    fn feedforward(&mut self, train_data: &DataBatch, print_out: bool);
    fn backpropagate(&mut self, train_data: &DataBatch);
    fn optimize_network(&mut self);

    fn error(&self) -> f32;

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>>;
    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>>;
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