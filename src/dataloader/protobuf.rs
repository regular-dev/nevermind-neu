use std::vec::Vec;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::cell::RefCell;

use crate::models::pb::{PbDataStorage, PbDataBatch};

use log::info;

use prost::Message;

use ndarray::Array;

use crate::dataloader::*;


#[derive(Default)]
pub struct ProtobufDataLoader {
    pub data: Vec<LabeledEntry>,
    pub id: RefCell<usize>,
}

impl ProtobufDataLoader {
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            id: RefCell::new(0),
        }
    }

    pub fn from_file(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut dl = ProtobufDataLoader::default();

        let buf = fs::read(filepath)?;
        let mut pb_data = PbDataStorage::decode(buf.as_slice())?;

        dl.data.reserve(pb_data.data.len());

        for i in pb_data.data.iter_mut() {
            let inp_vec = std::mem::replace(&mut i.input, Vec::new());
            let expected_vec = std::mem::replace(&mut i.expected, Vec::new());

            let input = Array::from_shape_vec(inp_vec.len(), inp_vec)?;
            let expected = Array::from_shape_vec(expected_vec.len(), expected_vec)?;

            dl.data.push(LabeledEntry{ 
                input: input,
                expected: expected
            });
        }

        Ok(dl)
    }

    pub fn to_file(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut pb_data = PbDataStorage::default();

        for i in &self.data {
            let inp_bor = i.input.clone(); // TODO : check

            let inp_vec = inp_bor.to_vec();
            let out_vec = i.expected.to_vec();

            pb_data.data.push(PbDataBatch{
                input: inp_vec,
                expected: out_vec
            });
        }

        let mut file = File::create(filepath)?;

        file.write_all(pb_data.encode_to_vec().as_slice())?;

        Ok(())
    }
}

impl DataLoader for ProtobufDataLoader {
    fn next(&self) -> &LabeledEntry {
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

        for _ in 0..size {
            mb.push(self.next());
        }

        MiniBatch::new(mb)
    }

    fn len(&self) -> Option< usize > {
        Some(self.data.len())
    }

    fn reset(&mut self) {
        *self.id.borrow_mut() = 0;
    }
}