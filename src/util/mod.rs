mod util;
mod normalize;
pub mod array_helpers;
pub mod activation;
#[cfg(feature = "opencl")]
pub mod activation_ocl;
pub mod with_params;

#[cfg(feature = "opencl")]
pub use activation_ocl::*;
pub use util::*;
pub use normalize::*;
pub use activation::*;
pub use with_params::*;