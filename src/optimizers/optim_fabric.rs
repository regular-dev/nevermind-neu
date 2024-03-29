use std::error::Error;
use std::fs::File;
use std::io::{Write, ErrorKind};

use log::{error, debug};

use crate::err::*;
use crate::optimizers::*;

pub fn optimizer_from_file(filepath: &str) -> Result<Box<dyn Optimizer>, Box<dyn Error>> {
    let cfg_file = File::open(filepath)?;
    let optim_params: SerdeWithParams = serde_yaml::from_reader(cfg_file)?;

    let optim_type = optim_params.0.get("type");

    if optim_type.is_none() {
        return Err(Box::new(CustomError::InvalidFormat));
    }

    if let Variant::String(optim_type) = optim_type.unwrap() {
        if optim_type == "rmsprop" {
            let mut rmsprop = Box::new(OptimizerRMS::default());
            rmsprop.set_cfg(&optim_params.0);

            debug!("Created rmsprop optimizer");

            return Ok(rmsprop);
        } else if optim_type == "sgd" {
            let mut sgd = Box::new(OptimizerSGD::default());
            sgd.set_cfg(&optim_params.0);

            debug!("Created sgd optimizer");

            return Ok(sgd);
        } else if optim_type == "adagrad" {
            let mut adagrad = Box::new(OptimizerAdaGrad::default());
            adagrad.set_cfg(&optim_params.0);

            debug!("Created adagrad optimizer");

            return Ok(adagrad);
        } else if optim_type == "adam" {
            let mut adam = Box::new(OptimizerAdam::default());
            adam.set_cfg(&optim_params.0);

            debug!("Created adam optimizer");

            return Ok(adam);
        } else {
            return Err(Box::new(CustomError::InvalidFormat));
        }
    }

    Err(Box::new(CustomError::InvalidFormat))
}

pub fn optimizer_to_file<T: Optimizer>(
    optimizer: T,
    filepath: &str,
) -> Result<(), Box<dyn Error>> {
    let mut helper = SerdeWithParams::default();

    let serde_params = optimizer.cfg();
    helper.0 = serde_params;

    let yaml_out_str = serde_yaml::to_string(&helper);
    let mut output = File::create(filepath)?;

    match yaml_out_str {
        Ok(yaml_str) => {
            output.write_all(yaml_str.as_bytes())?;
        },
        Err(x) => {
            error!("Error serializing optimizer");
            return Err(Box::new(std::io::Error::new(ErrorKind::Other, x)));
        }
    }

    Ok(())
}
