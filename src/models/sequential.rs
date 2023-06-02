use std::error::Error;
use std::ops::Deref;

use crate::layers_storage::SequentialLayersStorage;
use crate::models::Model;

use crate::models::pb::PbSequentialModel;
use crate::optimizers::{Optimizer, OptimizerRMS};

use std::fs::File;
use std::io::prelude::*;
use std::sync::Arc;
use std::{cell::RefCell, fs};

use log::{debug, error, info};
use std::io::ErrorKind;

use prost::Message;
use serde::{Deserialize, Deserializer, Serialize, *};

use crate::layer_fabric::*;
use crate::layers::*;
use crate::models::*;
use crate::util::*;

use super::pb::PbBufBlob;

#[derive(Clone)]
pub struct Sequential {
    ls: SequentialLayersStorage,
    batch_size: usize,
    optim: Box<dyn Optimizer>,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            ls: SequentialLayersStorage::empty(),
            batch_size: 1,
            optim: Box::new(OptimizerRMS::new(1e-2, 0.9)),
        }
    }

    pub fn new_simple(net_cfg: &Vec<usize>) -> Self {
        let mut seq = Self {
            ls: SequentialLayersStorage::new_simple_network(net_cfg),
            batch_size: 1,
            optim: Box::new(OptimizerRMS::new(1e-2, 0.9)),
        };
        seq.compile_shapes();

        return seq;
    }

    pub fn new_with_layers(ls: SequentialLayersStorage) -> Self {
        Self {
            ls,
            batch_size: 1,
            optim: Box::new(OptimizerRMS::new(1e-2, 0.9)),
        }
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

    pub fn compile_shapes(&mut self) {
        // TODO : may return some result in further
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

    pub fn set_optim(&mut self, optim: Box<dyn Optimizer>) {
        self.optim = optim;
    }

    pub fn add_layer(&mut self, l: Box<dyn AbstractLayer>) {
        self.ls.add_layer(l);
    }
}

impl Model for Sequential {
    fn feedforward(&mut self, train_data: Array2D) {
        let mut out = None;

        // for the first(input) layer
        {
            let l_first = self.ls.first_mut().unwrap();
            let result_out = l_first.forward_input(train_data);

            match result_out {
                Err(reason) => {
                    error!("[seq_mdl] {} error backpropagate : {}", l_first.layer_type(), reason);
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

    fn backpropagate(&mut self, expected: Array2D) {
        let expected_data = expected;

        // for the last layer
        {
            let prev_out = self.ls.at_mut(self.ls.len() - 2).cpu_params();
            let result_out = self
                .ls
                .at_mut(self.ls.len() - 1)
                .backward_output(vec![prev_out.unwrap()], expected_data);

            match result_out {
                Err(reason) => {
                    error!("[seq_mdl] Error backpropagate : {}", reason);
                    return;
                }
                Ok(_val) => {
                }
            }
        }

        for idx in 1..self.ls.len() {
            if idx == self.ls.len() - 1 {
                continue;
            }

            let prev_out = self.ls.at_mut(self.ls.len() - 2 - idx).cpu_params();
            let next_out = self.ls.at_mut(self.ls.len() - idx).cpu_params();

            let result_out = self
                .ls
                .at_mut(self.ls.len() - 1 - idx)
                .backward(vec![prev_out.unwrap()], vec![next_out.unwrap()]);

            match result_out {
                Err(reason) => {
                    error!("[seq_mdl] Error backpropagate : {}", reason);
                    return;
                }
                Ok(_val) => {
                }
            }
        }
    }

    fn optimize(&mut self) {
        for l in self.ls.iter_mut() {
            self.optim
                .optimize_params(&mut l.cpu_params().unwrap(), l.trainable_bufs());
        }
    }

    fn optimizer(&self) -> &Box<dyn WithParams> {
        // https://github.com/rust-lang/rust/issues/65991
        unsafe {
            let out = std::mem::transmute::<&Box<dyn Optimizer>, &Box<dyn WithParams>>(&self.optim);
            return out;
        }
    }

    fn optimizer_mut(&mut self) -> &mut Box<dyn WithParams> {
        // https://github.com/rust-lang/rust/issues/65991
        unsafe {
            let out = std::mem::transmute::<&mut Box<dyn Optimizer>, &mut Box<dyn WithParams>>(
                &mut self.optim,
            );
            return out;
        }
    }

    fn model_type(&self) -> &str {
        "mdl_sequential_cpu"
    }

    fn output_params(&self) -> CpuParams {
        let last_layer_params = self
            .ls
            .last()
            .expect("There is no layers in model !!!")
            .cpu_params()
            .unwrap();
        last_layer_params.clone()
    }

    fn last_layer_metrics(&self) -> Option<&Metrics> {
        self.last_layer().metrics()
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
        let mut vec_lr = Vec::with_capacity(self.ls.len());

        for l in self.ls.iter() {
            let mut pb_buf_blob = PbBufBlob::default(); // layer's buffers

            let bufs_ids = l.serializable_bufs();

            for i in bufs_ids.iter() {
                let buf = l.cpu_params().unwrap().get_param(*i);

                match buf {
                    VariantParamArc::Array2(arr2) => {
                        let arr2_bor = arr2.borrow();
                        let arr2_bor = arr2_bor.deref();

                        let pb_blob = model_helper::convert_buf_2d_to_pb(arr2_bor, *i);
                        pb_buf_blob.bufs.push(pb_blob);
                    }
                    VariantParamArc::Array1(arr1) => {
                        let arr1_bor = arr1.borrow();
                        let arr1_bor = arr1_bor.deref();

                        let pb_blob = model_helper::convert_buf_1d_to_pb(arr1_bor, *i);
                        pb_buf_blob.bufs.push(pb_blob);
                    }
                    _ => {
                        todo!()
                    }
                }
            }

            vec_lr.push(pb_buf_blob);
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

        for (self_l, l_pb) in self.ls.iter_mut().zip(&mut pb_model.layers) {
            if self_l.layer_type() == "InputLayer" {
                continue;
            }

            let mut layer_param = self_l.cpu_params().unwrap();

            for b_i in l_pb.bufs.iter() {
                layer_param.insert_buf(b_i.buf_id, model_helper::convert_pb_to_param_buf(&b_i))
            }

            self_l.set_cpu_params(layer_param);
        }

        Ok(())
    }
}

impl Serialize for Sequential {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq_mdl = SerdeSequentialModel::default();

        for l in self.ls.iter() {
            let s_layer_param = SerdeLayerParam {
                name: l.layer_type().to_owned(),
                params: l.cfg(),
            };
            seq_mdl.ls.push(s_layer_param);
        }

        seq_mdl.batch_size = self.batch_size();
        seq_mdl.mdl_type = self.model_type().to_string();

        seq_mdl.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Sequential {
    fn deserialize<D>(deserializer: D) -> Result<Sequential, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_mdl = SerdeSequentialModel::deserialize(deserializer)?;
        let mut seq_mdl = Sequential::new();

        if serde_mdl.mdl_type != seq_mdl.model_type() {
            todo!("Handle invalid model type on deserialization");
        }

        for i in serde_mdl.ls.iter() {
            let l_opt = create_layer(i.name.as_str(), Some(&i.params));

            if let Some(l) = l_opt {
                debug!("Create layer : {}", i.name);
                seq_mdl.add_layer(l);
            } else {
                // TODO : impl return D::Error
                panic!("Bad deserialization");
            }
        }

        seq_mdl.batch_size = serde_mdl.batch_size;

        Ok(seq_mdl)
    }
}
