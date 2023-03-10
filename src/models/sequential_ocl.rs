use crate::layers::*;
use crate::models::*;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use log::{debug, info};

use ocl::flags::{CommandQueueProperties, MapFlags, MemFlags};
use ocl::prm::Float4;
use ocl::{Buffer, Context, Device, Event, Kernel, Platform, Program, Queue, Result as OclResult};

use crate::layer_fabric::*;
use crate::layers_storage::*;

pub struct SequentialOcl {
    layers: Vec<Box<dyn AbstractLayerOcl>>,
    batch_size: usize,
    ocl_ctx: Context,
    ocl_queue: Queue,
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
            ocl_queue: kern_queue,
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

    pub fn add_layer(&mut self, l: Box<dyn AbstractLayerOcl>) {
        let mut l = l;
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
                ).expect("Input layer init ocl failure");
                continue;
            }

            l.init_ocl(
                &self.ocl_ctx,
                self.ocl_ctx.devices().first().unwrap().clone(),
                self.ocl_queue.clone(),
            )
            .expect("Init ocl failure");

            l.set_input_shape(&[prev_size]);
        }
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

    fn backpropagate(&mut self, expected: Batch) {}

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;

        for l in self.layers.iter_mut() {
            l.set_batch_size(self.batch_size);
        }
    }

    fn set_batch_size_for_tests(&mut self, batch_size: usize) {}

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

        for i in &self.layers {
            seq_mdl.layers.push(i.clone_layer_ocl());
        }

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
        let mut s_layers_storage = SerdeLayersStorage::default();

        for l in self.layers.iter() {
            let s_layer_param = SerdeLayerParam {
                name: l.layer_type().to_owned(),
                params: l.layer_cfg(),
            };
            s_layers_storage.layers_cfg.push(s_layer_param);
        }

        s_layers_storage.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SequentialOcl {
    fn deserialize<D>(deserializer: D) -> Result<SequentialOcl, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s_layers_storage = SerdeLayersStorage::deserialize(deserializer)?;

        let mut ls = SequentialOcl::new().unwrap();

        for i in &s_layers_storage.layers_cfg {
            let l_opt = create_layer_ocl(i.name.as_str(), Some(&i.params));

            if let Some(l) = l_opt {
                debug!("Create layer : {}", i.name);
                ls.layers.push(l);
            } else {
                // TODO : impl return D::Error
                panic!("Bad deserialization");
            }
        }

        Ok(ls)
    }
}
