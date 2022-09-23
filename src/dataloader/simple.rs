use crate::dataloader::DataBatch;
use crate::dataloader::DataLoader;

pub struct SimpleDataLoader {
    pub id: usize,
    pub data: Vec<DataBatch>,
}

impl DataLoader for SimpleDataLoader {
    fn next(&mut self) -> &DataBatch {
        assert!(self.data.len() > 0);

        if self.id < self.data.len() {
            let ret = &self.data[ self.id ];
            self.id += 1;
            return ret;
        } else {
            self.id = 0;
            return self.next();
        }
    }

    fn reset(&mut self) {
        self.id = 0;
    }
}

impl SimpleDataLoader {
    pub fn new(data: Vec<DataBatch>) -> Self {
        Self {
            id: 0,
            data,
        }
    }
}