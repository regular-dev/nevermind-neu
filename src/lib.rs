/// Folder
pub mod layers;
pub mod optimizers;
pub mod util;
pub mod dataloader;
pub mod models;

/// Files
pub mod layer_fabric;
pub mod layers_storage;
pub mod cpu_params;
pub mod orchestra;
pub mod err;
#[cfg(feature = "opencl")]
pub mod ocl;

pub mod prelude {
    pub use crate::orchestra::save_model_cfg;
    // pub use crate::network::* and etc...
}