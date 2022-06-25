use crate::mind::abstract_layer::Blob;

#[derive(Clone)]
pub struct DataBatch {
    pub input: Blob,
    pub expected: Blob,
}

impl DataBatch {
    pub fn new(input: Vec<f32>, expected: Vec<f32>) -> Self {
        Self {
            input: vec![input],
            expected: vec![expected]
        }
    }
}

pub trait DataLoader {
    fn next(&mut self) -> &DataBatch;
    fn reset(&mut self) { }
}

pub struct SimpleDataLoader {
    pub cur_idx: usize,
    pub data: Vec<DataBatch>,
}

impl DataLoader for SimpleDataLoader {
    fn next(&mut self) -> &DataBatch {
        if self.cur_idx < self.data.len() {
            let ret = &self.data[ self.cur_idx ];
            self.cur_idx += 1;
            return ret;
        } else {
            self.cur_idx = 0;
            return self.next();
        }
    }
}

impl SimpleDataLoader {
    pub fn new(data: Vec<DataBatch>) -> Self {
        Self {
            cur_idx: 0,
            data,
        }
    }
}