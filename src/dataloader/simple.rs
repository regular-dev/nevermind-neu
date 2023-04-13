use std::{cell::RefCell, error::Error, fs::File};

use crate::dataloader::{DataLoader, LabeledEntry, MiniBatch};

pub struct SimpleDataLoader {
    pub id: RefCell<usize>,
    pub data: Vec<LabeledEntry>,
}

impl DataLoader for SimpleDataLoader {
    fn next(&self) -> &LabeledEntry {
        assert!(self.data.len() > 0);

        let mut self_id = self.id.borrow_mut();

        if *self_id < self.data.len() {
            let ret = &self.data[*self_id];
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

    fn len(&self) -> Option< usize > {
        Some(self.data.len())
    }

    fn pos(&self) -> Option< usize > {
        Some(*self.id.borrow())
    }
}

impl SimpleDataLoader {
    pub fn new(data: Vec<LabeledEntry>) -> Self {
        Self {
            id: RefCell::new(0),
            data,
        }
    }

    pub fn from_csv_file(filepath: &str, lbl_col_count: usize) -> Result<Self, Box<dyn Error>> {
        let file = File::open(filepath)?;
        let mut rdr = csv::Reader::from_reader(file);

        let records = rdr.records();

        let mut data = Vec::new();

        for row in records {
            let row = row?;

            let inp_len = row.len() - lbl_col_count;

            let mut inp_vec = Vec::with_capacity(inp_len);
            let mut out_vec = Vec::with_capacity(lbl_col_count);

            for (idx, val) in row.iter().enumerate() {
                if idx > inp_len {
                    break;
                }

                inp_vec.push(val.parse::<f32>()?);
            }

            for val in row.iter().skip(inp_len) {
                out_vec.push(val.parse::<f32>()?);
            }

            let lbl_entry = LabeledEntry::new(inp_vec, out_vec);
            data.push(lbl_entry);
        }

        Ok(SimpleDataLoader::new(data))
    }

    pub fn empty() -> Self {
        Self {
            id: RefCell::new(0),
            data: vec![],
        }
    }
}
