use std::cell::RefCell;

use crate::dataloader::{DataBatch, MiniBatch, DataLoader};

pub struct SimpleDataLoader {
    pub id: RefCell<usize>,
    pub data: Vec<DataBatch>,
}

impl DataLoader for SimpleDataLoader {
    fn next(&self) -> &DataBatch {
        assert!(self.data.len() > 0);

        let mut self_id = self.id.borrow_mut();

        if *self_id < self.data.len() {
            let ret = &self.data[ *self_id ];
            *self_id += 1;
            return ret;
        } else {
            *self_id = 0;
            drop(self_id);

            return self.next();
        }
    }

    fn next_batch(&self, size: usize) -> MiniBatch {
        let mut mb = Vec::with_capacity(size);

        for _i in 0..size {
            mb.push(self.next());
        }

        MiniBatch::new(mb)
    }

    fn reset(&mut self) {
        *self.id.borrow_mut() = 0;
    }
}

impl SimpleDataLoader {
    pub fn new(data: Vec<DataBatch>) -> Self {
        Self {
            id: RefCell::new(1),
            data,
        }
    }
}