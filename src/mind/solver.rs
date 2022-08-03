use super::dataset::DataBatch;
use super::layers_storage::LayersStorage;

pub trait Solver {
    fn setup_network(&mut self, layers: LayersStorage);
    fn perform_step(&mut self, data: &DataBatch);
    fn feedforward(&mut self, train_data: &DataBatch, print_out: bool);
    fn backpropagate(&mut self, train_data: &DataBatch);
    fn optimize_network(&mut self);
}