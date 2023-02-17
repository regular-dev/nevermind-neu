use crate::dataloader::databatch::{DataBatch, MiniBatch};

pub trait DataLoader {
    fn next(&self) -> &DataBatch;
    fn next_batch(&self, size: usize) -> MiniBatch;

    fn reset(&mut self) { }
    fn len(&self) -> Option< usize > { None }
}
