use std::{collections::HashMap, hash::Hash};
use crate::util::*;

use serde::{Serialize, Deserialize};

pub trait WithParams {
    fn cfg(&self) -> HashMap<String, Variant> { HashMap::new() }
    fn set_cfg(&mut self, args: &HashMap<String, Variant>) { }
}

#[derive(Serialize, Deserialize, Default)]
pub struct SerdeWithParams( pub HashMap<String, Variant>); 