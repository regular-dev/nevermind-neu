use crate::layers::*;
use crate::models::*;

//#[derive(Clone)]
pub struct SequentialOcl {
    ls: Vec<Box<dyn AbstractLayerOcl>>,
    batch_size: usize,
}

impl SequentialOcl {
    pub fn new() -> Self {
        Self {
            ls: Vec::new(),
            batch_size: 1,
        }
    }
}

impl Model for SequentialOcl {
    fn feedforward(&mut self, train_data: Batch) {}

    fn backpropagate(&mut self, expected: Batch) {}

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    fn set_batch_size_for_tests(&mut self, batch_size: usize) {}

    // TODO : maybe make return value Option<...>
    fn layer(&self, id: usize) -> &Box<dyn AbstractLayer> {
        todo!()
    }
    fn layers_count(&self) -> usize {
        self.ls.len()
    }
    fn last_layer(&self) -> &Box<dyn AbstractLayer> {
        todo!()
    }

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>> {
        todo!()
    }
    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>> {
        todo!()
    }
}
