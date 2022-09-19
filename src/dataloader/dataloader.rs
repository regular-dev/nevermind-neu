use crate::dataloader::databatch::DataBatch;

pub trait DataLoader {
    fn next(&mut self) -> &DataBatch;
    fn reset(&mut self) { }
}
