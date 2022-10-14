use std::vec::Vec;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::cell::RefCell;

use crate::solvers::pb::{PbDataStorage, PbDataBatch};

use prost::Message;

use ndarray::Array;

use crate::dataloader::*;
use crate::util::DataVecPtr;


#[derive(Default)]
pub struct ProtobufDataLoader {
    pub data: Vec<DataBatch>,
    pub id: usize,
}

impl ProtobufDataLoader {
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            id: 0,
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

            dl.data.push(DataBatch{ 
                input: DataVecPtr::new(RefCell::new(input)),
                expected: expected
            });
        }

        Ok(dl)
    }

    pub fn to_file(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut pb_data = PbDataStorage::default();

        for i in &self.data {
            let inp_bor = i.input.borrow();

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
    fn next(&mut self) -> &DataBatch {
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