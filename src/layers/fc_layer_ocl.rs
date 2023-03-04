use crate::learn_params::LearnParams;
use crate::layers::abstract_layer_ocl::*;

pub struct FcLayerOcl {
    cpu_params: LearnParams,
    gpu_params: OclParams,
}