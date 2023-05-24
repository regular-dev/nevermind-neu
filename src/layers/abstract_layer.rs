use std::collections::HashMap;

use crate::learn_params::{LearnParams, ParamsBlob};
use crate::util::{Array2D, WithParams, Metrics};

#[derive(Debug)]
pub enum LayerError {
    InvalidSize,
    OtherError,
    NotImpl,
}

pub type LayerForwardResult = Result<ParamsBlob, LayerError>;
pub type LayerBackwardResult = Result<ParamsBlob, LayerError>;

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
        let mut lr = self.learn_params().unwrap();
        lr.fit_to_batch_size(batch_size);
    }

    fn metrics(&self) -> Option<&Metrics> {
        None
    }

    fn learn_params(&self) -> Option<LearnParams>;
    fn set_learn_params(&mut self, lp: LearnParams);

    fn set_input_shape(&mut self, sh: &[usize]);

    // Do copy layer memory(ws, output, ...)
    fn copy_layer(&self) -> Box<dyn AbstractLayer>;

    // Do copy only Rc
    fn clone_layer(&self) -> Box<dyn AbstractLayer>;
}
