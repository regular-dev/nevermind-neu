use std::fmt;

use crate::cpu_params::{CpuParams, ParamsBlob, TypeBuffer};
use crate::util::{Array2D, Metrics, WithParams};

#[derive(Debug)]
pub enum LayerError {
    InvalidSize,
    OtherError,
    NotImpl,
}

impl fmt::Display for LayerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LayerError::InvalidSize => {
                write!(f, "{}", "Invalid size")
            },
            LayerError::OtherError => {
                write!(f, "{}", "OtherError")
            }
            _ => {
                write!(f, "{}", "Other")
            },
        }
    }
}


pub type LayerForwardResult = Result<ParamsBlob, LayerError>;
pub type LayerBackwardResult = Result<ParamsBlob, LayerError>;
pub type TrainableBufsIds<'a> = (&'a[i32], &'a[i32]);

pub trait AbstractLayer: WithParams {
    // for signature for input layers
    fn forward_input(&mut self, _input_data: Array2D) -> LayerForwardResult {
        Err(LayerError::NotImpl)
    }

    fn forward(&mut self, _input: ParamsBlob) -> LayerForwardResult {
        Err(LayerError::NotImpl)
    }

    /// returns out_values and array of weights
    fn backward(&mut self, _prev_input: ParamsBlob, _input: ParamsBlob) -> LayerBackwardResult {
        Err(LayerError::NotImpl)
    }

    fn backward_output(
        &mut self,
        _prev_input: ParamsBlob,
        _expected: Array2D,
    ) -> LayerBackwardResult {
        Err(LayerError::NotImpl)
    }

    fn layer_type(&self) -> &str;

    fn size(&self) -> usize;

    fn set_batch_size(&mut self, batch_size: usize) {
        let mut lr = self.cpu_params().unwrap();
        lr.fit_to_batch_size(batch_size);
    }

    fn metrics(&self) -> Option<&Metrics> {
        None
    }

    fn serializable_bufs(&self) -> &[i32] {
        return &[TypeBuffer::Weights as i32, TypeBuffer::Bias as i32];
    }

    fn trainable_bufs(&self) -> TrainableBufsIds {
        (
            &[TypeBuffer::Weights as i32, TypeBuffer::Bias as i32],
            &[TypeBuffer::WeightsGrad as i32, TypeBuffer::BiasGrad as i32],
        )
    }

    fn cpu_params(&self) -> Option<CpuParams>;
    fn set_cpu_params(&mut self, lp: CpuParams);

    fn set_input_shape(&mut self, sh: &[usize]);

    // Do copy layer memory(ws, output, ...)
    fn copy_layer(&self) -> Box<dyn AbstractLayer>;

    // Do copy only Rc
    fn clone_layer(&self) -> Box<dyn AbstractLayer>;
}
