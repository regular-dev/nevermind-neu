use std::error::Error;
use std::ops::Deref;

use crate::layers_storage::SequentialLayersStorage;
use crate::models::Model;

use crate::models::pb::PbSequentialModel;

use std::fs;
use std::fs::File;
use std::io::prelude::*;

use log::error;
use std::io::ErrorKind;

use prost::Message;
use serde::{Deserialize, Serialize};

use crate::layers::*;
use crate::models::*;
use crate::util::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct Sequential {
    ls: SequentialLayersStorage,
    batch_size: usize,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            ls: SequentialLayersStorage::empty(),
            batch_size: 1,
        }
    }

    pub fn new_simple(net_cfg: &Vec<usize>) -> Self {
        let mut seq = Self {
            ls: SequentialLayersStorage::new_simple_network(net_cfg),
            batch_size: 1,
        };
        seq.compile_shapes();

        return seq;
    }

    pub fn new_with_layers(ls: SequentialLayersStorage) -> Self {
        Self { ls, batch_size: 1 }
    }

    pub fn from_file(filepath: &str) -> Result<Self, Box<dyn Error>> {
        let cfg_file = File::open(filepath)?;
        let mut mdl: Sequential = serde_yaml::from_reader(cfg_file)?;
        mdl.compile_shapes();
        mdl.set_batch_size(mdl.batch_size);

        Ok(mdl)
    }

    pub fn to_file(&self, filepath: &str) -> Result<(), Box<dyn Error>> {
        let yaml_str_result = serde_yaml::to_string(&self);

        let mut output = File::create(filepath)?;

        match yaml_str_result {
            Ok(yaml_str) => {
                output.write_all(yaml_str.as_bytes())?;
            }
            Err(x) => {
                error!("Error (serde-yaml) serializing net layers !!!");
                return Err(Box::new(std::io::Error::new(ErrorKind::Other, x)));
            }
        }

        Ok(())
    }

    pub fn compile_shapes(&mut self) { // TODO : may return some result in further
        let mut prev_size = 0;

        for (idx, l) in self.ls.iter_mut().enumerate() {
            if idx == 0 {
                prev_size = l.size();
                continue;
            }

            l.set_input_shape(&[prev_size]);
            prev_size = l.size();
        }
    }
}

impl Model for Sequential {
    fn feedforward(&mut self, train_data: Batch) {
        let mut out = None;

        // for the first(input) layer
        {
            let l_first = self.ls.first_mut().unwrap();
            let result_out = l_first.forward_input(train_data);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            }
        }

        for l in self.ls.iter_mut().skip(1) {
            let result_out = l.forward(out.unwrap());

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            };
        }
    }

    fn backpropagate(&mut self, expected: Batch) {
        let expected_data = expected;

        let mut out = None;

        // for the last layer
        {
            let prev_out = self.ls.at_mut(self.ls.len() - 2).learn_params();
            let result_out = self
                .ls
                .at_mut(self.ls.len() - 1)
                .backward_output(vec![prev_out.unwrap()], expected_data);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            }
        }

        for idx in 1..self.ls.len() {
            if idx == self.ls.len() - 1 {
                continue;
            }

            let prev_out = self.ls.at_mut(self.ls.len() - 2 - idx).learn_params();
            let next_out = self.ls.at_mut(self.ls.len() - idx).learn_params();

            let result_out = self
                .ls
                .at_mut(self.ls.len() - 1 - idx)
                .backward(vec![prev_out.unwrap()], vec![next_out.unwrap()]);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            }
        }
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
        self.ls.fit_to_batch_size(self.batch_size);
    }

    fn set_batch_size_for_tests(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
        self.ls.prepare_for_tests(batch_size);
    }

    fn layer(&self, id: usize) -> &Box<dyn AbstractLayer> {
        self.ls.at(id)
    }

    fn layers_count(&self) -> usize {
        self.ls.len()
    }

    fn last_layer(&self) -> &Box<dyn AbstractLayer> {
        self.ls.last().unwrap()
    }

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        // create vector of layers learn_params
        let mut vec_lr = Vec::new();
        for l in 0..self.ls.len() {
            let lr_params = self.ls.at(l).learn_params().unwrap();
            let ws = lr_params.ws.borrow();
            vec_lr.push(model_helper::convert_ws_blob_to_pb(ws.deref()));
        }

        let pb_model = PbSequentialModel { layers: vec_lr };

        // encode
        let mut file = File::create(filepath)?;

        file.write_all(pb_model.encode_to_vec().as_slice())?;

        Ok(())
    }

    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let buf = fs::read(filepath)?;

        let mut pb_model = PbSequentialModel::decode(buf.as_slice())?;

        for (self_l, l) in self.ls.iter_mut().zip(&mut pb_model.layers) {
            let layer_param = self_l.learn_params().unwrap();
            let mut l_ws = layer_param.ws.borrow_mut();
            *l_ws = model_helper::convert_pb_to_ws_blob(l);
        }

        Ok(())
    }
}
