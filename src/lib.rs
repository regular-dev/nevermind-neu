/// Folder
pub mod layers;
pub mod optimizers;
pub mod util;
pub mod dataloader;
pub mod models;

/// Files
pub mod activation;
pub mod layer_fabric;
pub mod layers_storage;
pub mod learn_params;
pub mod network;
pub mod err;

pub mod prelude {
    pub use crate::network::save_model_cfg;
    // pub use crate::network::* and etc...
}