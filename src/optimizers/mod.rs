mod optim_sgd;
mod optim_adagrad;
mod optim_rms;
mod optim_adam;

#[cfg(feature = "opencl")]
mod optim_ocl_sgd;
#[cfg(feature = "opencl")]
mod optim_ocl_rms;
#[cfg(feature = "opencl")]
mod optim_ocl_adam;
#[cfg(feature = "opencl")]
mod optim_ocl;
#[cfg(feature = "opencl")]
mod optim_ocl_fabric;

mod optim_fabric;

pub use optim_rms::*;
pub use optim_adagrad::*;
pub use optim_adam::*;
pub use optim_sgd::*;
pub use optim_fabric::*;
#[cfg(feature = "opencl")]
pub use optim_ocl_sgd::*;
#[cfg(feature = "opencl")]
pub use optim_ocl_rms::*;
#[cfg(feature = "opencl")]
pub use optim_ocl_adam::*;
#[cfg(feature = "opencl")]
pub use optim_ocl::*;
#[cfg(feature ="opencl")]
pub use optim_ocl_fabric::*;

use crate::learn_params::*;
use crate::util::*;
use crate::err::*;

pub trait Optimizer : WithParams {
    fn optimize_params(&mut self, learn_params: &mut LearnParams);
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

impl Default for Box<dyn Optimizer> {
    fn default() -> Self {
        Box::new(OptimizerRMS::new(1e-2, 0.9))
    }
}

impl Clone for Box<dyn Optimizer> {
    fn clone(&self) -> Self {
        Box::new(OptimizerRMS::new(1e-2, 0.9))
    }
}
