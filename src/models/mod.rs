mod sequential;
mod model_helper;

use std::error::Error;
use crate::{util::Batch, layers::AbstractLayer};

pub use sequential::*;

pub trait Model {
    fn feedforward(&mut self, train_data: Batch);
    fn backpropagate(&mut self, expected: Batch);
    fn batch_size(&self) -> usize;
    fn set_batch_size(&mut self, batch_size: usize);
    fn set_batch_size_for_tests(&mut self, batch_size: usize);
    
    // TODO : maybe make return value Option<...>
    fn layer(&self, id: usize) -> &Box<dyn AbstractLayer>;
    fn layers_count(&self) -> usize;
    fn last_layer(&self) -> &Box<dyn AbstractLayer >; 

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>>;
    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>>;
}

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/mind.serial_pb.rs"));
}
