mod sequential;
mod model_helper;
#[cfg(feature = "opencl")]
mod sequential_ocl;

use std::{error::Error, rc::Rc, cell::RefCell};
use crate::{util::*, layers::AbstractLayer, learn_params::*, layers_storage::SerdeLayersStorage};
use crate::layers_storage::*;

pub use sequential::*;
#[cfg(feature = "opencl")]
pub use sequential_ocl::*;
use serde::{Serialize, Deserialize};

pub trait Model {
    fn feedforward(&mut self, train_data: Batch);
    fn backpropagate(&mut self, expected: Batch);
    fn optimize(&mut self);
    fn batch_size(&self) -> usize;
    fn set_batch_size(&mut self, batch_size: usize);
    fn set_batch_size_for_tests(&mut self, batch_size: usize);
    
    // TODO : maybe make return value Option<...>
    fn layer(&self, id: usize) -> &Box<dyn AbstractLayer>;
    fn layers_count(&self) -> usize;
    fn last_layer(&self) -> &Box<dyn AbstractLayer >; 

    fn output_params(&self) -> LearnParams;

    fn model_type(&self) -> &str;

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>>;
    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>>;
}

#[derive(Serialize, Deserialize)]
pub struct SerdeSequentialModel {
    pub ls: Vec<SerdeLayerParam>,
    pub mdl_type: String,
    pub batch_size: usize,
}

impl Default for SerdeSequentialModel {
    fn default() -> Self {
        SerdeSequentialModel {
            ls: Vec::new(),
            mdl_type: "none".to_string(),
            batch_size: 1,
        }
    }
}

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/mind.serial_pb.rs"));
}
