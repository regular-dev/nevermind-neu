use std::{error::Error, fs::File, io::prelude::*, io::ErrorKind};

use crate::layers::TrainableBufsIds;
use crate::ocl::*;
use crate::util::*;

pub trait OptimizerOcl: WithParams {
    fn optimize_ocl_params(&mut self, params: OclParams, train_bufs: TrainableBufsIds);
}
