use serde::{Serialize, Serializer};
use serde::ser::{SerializeSeq};

use super::abstract_layer::AbstractLayer;


pub struct LayersStorage {
    layers: Vec<Box<dyn AbstractLayer>>,
}

impl LayersStorage {
    pub fn new() -> Self {
        LayersStorage { layers: Vec::new() }
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Box<dyn AbstractLayer>> {
        return self.layers.iter_mut();
    }

    pub fn add_layer(&mut self, l: Box<dyn AbstractLayer>) {
        self.layers.push(l);
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn at(&mut self, id: usize) -> &mut Box<dyn AbstractLayer> {
        &mut self.layers[id]
    }
}

impl Serialize for LayersStorage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut layers_cfg = serializer.serialize_seq(Some(self.layers.len()))?;
        for l in self.layers.iter() {
            let l_cfg = l.layer_cfg();
            layers_cfg.serialize_element(&l_cfg)?;
        }

        layers_cfg.end()
    }
}