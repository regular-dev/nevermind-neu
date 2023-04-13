use crate::dataloader::databatch::{LabeledEntry, MiniBatch};

pub trait DataLoader {
    fn next(&self) -> &LabeledEntry;
    fn next_batch(&self, size: usize) -> MiniBatch;

    fn reset(&mut self) { }
    fn len(&self) -> Option< usize > { None }
    fn pos(&self) -> Option< usize > { None }
}
