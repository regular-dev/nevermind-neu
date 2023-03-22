use std::{cell::RefCell, fmt, fs::File, io::prelude::*, io::ErrorKind, rc::Rc};

use crate::layers::*;
use crate::learn_params::LearnParams;
use crate::models::*;
use crate::optimizers::*;
use crate::util::*;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use log::{debug, error, info};

use ocl::flags::{CommandQueueProperties, MapFlags, MemFlags};
use ocl::{Buffer, Context, Device, Event, Kernel, Platform, Program, Queue, Result as OclResult};

use crate::layer_fabric::*;
use crate::layers_storage::*;

pub struct SequentialOcl {
    layers: Vec<Box<dyn AbstractLayerOcl>>,
    batch_size: usize,
    ocl_ctx: Context,
    ocl_queue: Queue,
    optim: Box<dyn OptimizerOcl>,
}

impl SequentialOcl {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let platform = Platform::default();
        info!("[OCL] Platform is {}", platform.name()?);

        let device = Device::first(&platform)?;
        info!("[OCL] Device is {} - {}", device.vendor()?, device.name()?);

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        let kern_queue = Queue::new(&context, device, None)?;

        Ok(Self {
            layers: Vec::new(),
            batch_size: 1,
            ocl_ctx: context,
            ocl_queue: kern_queue.clone(),
            optim: Box::new(OptimizerOclRms::new(kern_queue.clone())),
        })
    }

    pub fn new_simple(net_cfg: &Vec<usize>) -> Self {
        let mut mdl = SequentialOcl::new().unwrap();

        for (idx, i) in net_cfg.iter().enumerate() {
            if idx == 0 {
                mdl.add_layer(Box::new(InputLayerOcl::new(*i)));
                continue;
            }

            if idx == net_cfg.len() - 1 {
                mdl.add_layer(Box::new(EuclideanLossLayerOcl::new(*i)));
                continue;
            }

            mdl.add_layer(Box::new(FcLayerOcl::new(*i)));
        }

        mdl.init_layers();

        mdl
    }

    pub fn from_file(filepath: &str) -> Result<Self, Box<dyn Error>> {
        let cfg_file = File::open(filepath)?;
        let mut mdl: SequentialOcl = serde_yaml::from_reader(cfg_file)?;        
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

    pub fn add_layer(&mut self, l: Box<dyn AbstractLayerOcl>) {
        self.layers.push(l);
    }

    pub fn init_layers(&mut self) {
        let mut prev_size = 0;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            if idx == 0 {
                prev_size = l.size();
                l.init_ocl(
                    &self.ocl_ctx,
                    self.ocl_ctx.devices().first().unwrap().clone(),
                    self.ocl_queue.clone(),
                )
                .expect("Input layer init ocl failure");
                continue;
            }

            l.init_ocl(
                &self.ocl_ctx,
                self.ocl_ctx.devices().first().unwrap().clone(),
                self.ocl_queue.clone(),
            )
            .expect("Init ocl failure");

            l.set_input_shape(&[prev_size]);

            prev_size = l.size();
        }
    }

    fn init_layers_but_weights(&mut self) {
        let mut prev_size = 0;

        for (idx, l) in self.layers.iter_mut().enumerate() {
            if idx == 0 {
                prev_size = l.size();
                l.init_ocl(
                    &self.ocl_ctx,
                    self.ocl_ctx.devices().first().unwrap().clone(),
                    self.ocl_queue.clone(),
                )
                .expect("Input layer init ocl failure");
                continue;
            }

            l.init_ocl(
                &self.ocl_ctx,
                self.ocl_ctx.devices().first().unwrap().clone(),
                self.ocl_queue.clone(),
            )
            .expect("Init ocl failure");

            // We need to copy weights buffer cause of
            // https://registry.khronos.org/OpenCL/sdk/1.2/docs/man/xhtml/clSetKernelArg.html#notes
            let train_mdl_params = l.ocl_params().unwrap();
            let train_mdl_ws = train_mdl_params.ws.borrow();
            let mut vec_ws = vec![0.0; train_mdl_ws.len()];

            train_mdl_ws
                .read(&mut vec_ws)
                .enq()
                .expect("Failed to read train model weights");

            l.set_input_shape(&[prev_size]);

            let mut new_mdl_params = l.ocl_params().unwrap();
            let new_mdl_ws = Rc::new(RefCell::new(
                Buffer::builder()
                    .queue(self.ocl_queue.clone())
                    .flags(MemFlags::new().read_write())
                    .len(vec_ws.len())
                    .copy_host_slice(vec_ws.as_slice())
                    .build()
                    .expect("Failed to copy WS buffer"),
            ));

            new_mdl_params.ws = new_mdl_ws;

            l.set_ocl_params(new_mdl_params);

            prev_size = l.size();
        }
    }

    pub fn set_optim(&mut self, opt: Box<dyn OptimizerOcl>) {
        self.optim = opt;
    }

    pub fn queue(&self) -> Queue {
        self.ocl_queue.clone()
    }
}

impl Model for SequentialOcl {
    fn feedforward(&mut self, train_data: Batch) {
        let mut out = None;

        // for the first(input) layer
        {
            let l_first = self.layers.first_mut().unwrap();
            let result_out = l_first.forward_input_ocl(train_data);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            }
        }

        for l in self.layers.iter_mut().skip(1) {
            let result_out = l.forward_ocl(out.unwrap());

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
        let mut out = None;

        let layers_len = self.layers.len();

        // for the last layer
        {
            let prev_out = self.layers[layers_len - 2].ocl_params();

            let last_layer_idx = layers_len - 1;

            let result_out =
                self.layers[last_layer_idx].backward_output_ocl(vec![prev_out.unwrap()], expected);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            }
        }

        // TODO : refactor below
        for idx in 1..layers_len {
            if idx == layers_len - 1 {
                continue;
            }

            let prev_out = self.layers[layers_len - 2 - idx].ocl_params();
            let next_out = self.layers[layers_len - idx].ocl_params();

            let layer_idx = layers_len - 1 - idx;

            let result_out = self.layers[layer_idx]
                .backward_ocl(vec![prev_out.unwrap()], vec![next_out.unwrap()]);

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

    fn optimize(&mut self) {
        for l in self.layers.iter_mut().skip(1) {
            self.optim.optimize_ocl_params(l.ocl_params().unwrap());
        }
    }

    fn model_type(&self) -> &str {
        "mdl_sequential_ocl"
    }

    fn output_params(&self) -> LearnParams {
        let ocl_params = self
            .layers
            .last()
            .expect("There are no layers in model ocl !!!")
            .ocl_params()
            .unwrap();

        let cpu_lp = self
            .layers
            .last()
            .expect("There are no layers in ocl model")
            .learn_params()
            .unwrap();
        let mut cpu_output = cpu_lp.output.borrow_mut();
        let mut cpu_neu_grad = cpu_lp.err_vals.borrow_mut();

        // Fetch data from OCL Buffer to cpu memory
        let ocl_params_output = ocl_params.output.borrow();
        ocl_params_output
            .read(cpu_output.as_slice_mut().unwrap())
            .enq()
            .expect("Failed to copy OCL buffer to CPU");

        let ocl_params_neu_grad = ocl_params.neu_grad.borrow();
        ocl_params_neu_grad
            .read(cpu_neu_grad.as_slice_mut().unwrap())
            .enq()
            .expect("Failed top copy OCL buffer to CPU");

        return cpu_lp.clone();
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;

        for l in self.layers.iter_mut() {
            l.set_batch_size(self.batch_size);
        }
    }

    fn set_batch_size_for_tests(&mut self, batch_size: usize) {
        self.batch_size = batch_size;

        for l in self.layers.iter_mut() {
            l.set_batch_size(self.batch_size);
        }
    }

    // TODO : maybe make return value Option<...>
    fn layer(&self, id: usize) -> &Box<dyn AbstractLayer> {
        todo!()
    }
    fn layers_count(&self) -> usize {
        self.layers.len()
    }
    fn last_layer(&self) -> &Box<dyn AbstractLayer> {
        todo!()
    }

    fn save_state(&self, filepath: &str) -> Result<(), Box<dyn Error>> {
        todo!()
    }
    fn load_state(&mut self, filepath: &str) -> Result<(), Box<dyn Error>> {
        todo!()
    }
}

impl Clone for SequentialOcl {
    fn clone(&self) -> Self {
        let mut seq_mdl = SequentialOcl::new().unwrap();

        for i in self.layers.iter() {
            seq_mdl.layers.push(i.clone_layer_ocl());
        }

        seq_mdl.init_layers_but_weights();
        seq_mdl.set_batch_size(self.batch_size);

        seq_mdl
    }

    fn clone_from(&mut self, source: &Self) {
        todo!()
    }
}

impl Serialize for SequentialOcl {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq_mdl = SerdeSequentialModel::default();

        for l in self.layers.iter() {
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

impl<'de> Deserialize<'de> for SequentialOcl {
    fn deserialize<D>(deserializer: D) -> Result<SequentialOcl, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_mdl = SerdeSequentialModel::deserialize(deserializer)?;

        let mut seq_mdl = SequentialOcl::new().expect("Failed to create SequentialOcl model");

        if serde_mdl.mdl_type != seq_mdl.model_type() {
            todo!("Handle invalid model type");
        }

        for i in &serde_mdl.ls {
            let l_opt = create_layer_ocl(i.name.as_str(), Some(&i.params));

            if let Some(l) = l_opt {
                debug!("Create layer : {}", i.name);
                seq_mdl.layers.push(l);
            } else {
                // TODO : impl return D::Error
                panic!("Bad deserialization");
            }
        }

        seq_mdl.init_layers();
        seq_mdl.set_batch_size(serde_mdl.batch_size);

        Ok(seq_mdl)
    }
}

impl fmt::Display for SequentialOcl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut out = String::new();

        for i in &self.layers {
            out += i.size().to_string().as_str();
            out += "-";
        }

        write!(f, "{}", &out.as_str()[0..out.len() - 1])
    }
}
