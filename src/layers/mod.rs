mod abstract_layer;
mod dummy_layer;
mod error_layer;
mod softmax_loss_layer;
mod fc_layer;
mod input_data_layer;

#[cfg(feature = "opencl")]
mod abstract_layer_ocl;
#[cfg(feature = "opencl")]
mod fc_layer_ocl;

pub use abstract_layer::*;
pub use dummy_layer::*;
pub use error_layer::*;
pub use fc_layer::*;
pub use input_data_layer::*;
pub use softmax_loss_layer::*;