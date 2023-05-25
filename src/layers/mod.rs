mod abstract_layer;
mod dummy_layer;
mod euclidean_loss_layer;
mod softmax_loss_layer;
mod fc_layer;
mod input_layer;

#[cfg(feature = "opencl")]
mod abstract_layer_ocl;
#[cfg(feature = "opencl")]
mod fc_layer_ocl;
#[cfg(feature = "opencl")]
mod input_layer_ocl;
#[cfg(feature = "opencl")]
mod euclidean_loss_layer_ocl;
#[cfg(feature = "opencl")]
mod softmax_loss_layer_ocl;

pub use abstract_layer::*;
pub use dummy_layer::*;
pub use euclidean_loss_layer::*;
pub use fc_layer::*;
pub use input_layer::*;
pub use softmax_loss_layer::*;
#[cfg(feature = "opencl")]
pub use abstract_layer_ocl::*;
#[cfg(feature = "opencl")]
pub use input_layer_ocl::*;
#[cfg(feature = "opencl")]
pub use fc_layer_ocl::*;
#[cfg(feature = "opencl")]
pub use euclidean_loss_layer_ocl::*;
#[cfg(feature = "opencl")]
pub use softmax_loss_layer_ocl::*;