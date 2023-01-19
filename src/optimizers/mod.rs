mod optim_sgd;
mod optim_adagrad;
mod optim_rms;
mod optim_adam;

mod optim_creator;

use std::collections::HashMap;

pub use optim_rms::*;
pub use optim_adagrad::*;
pub use optim_adam::*;
pub use optim_sgd::*;
pub use optim_creator::*;

use crate::learn_params::LearnParams;
use crate::util::*;
use crate::err::*;

pub trait Optimizer {
    fn optimize_network(&mut self, learn_params: &mut LearnParams);
    
    fn cfg(&self) -> HashMap<String, Variant>;
    fn set_cfg(&mut self, args: &HashMap<String, Variant>);
}

pub fn optimizer_from_type(opt_type: &str) -> Result<Box<dyn Optimizer>, CustomError> {
    match opt_type {
        "rmsprop" => {
            return Ok(Box::new(OptimizerRMS::default()));
        },
        "sgd" => {
            return Ok(Box::new(OptimizerSGD::default()));
        },
        "adagrad" => {
            return Ok(Box::new(OptimizerAdaGrad::default()));
        },
        "adam" => {
            return Ok(Box::new(OptimizerAdam::default()));
        },
        _ => {
            return Err(CustomError::WrongArg);
        }
    }
}