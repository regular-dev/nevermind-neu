use std::option::Option;

use super::learn_params::LearnParams;
use super::util::{Blob, DataVec, Variant, WsBlob, WsMat};
use crate::mind::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};

// not used
pub struct DummyLayer {
    output: DataVec,
    fake_lr: LearnParams,
}

impl AbstractLayer for DummyLayer {
    fn forward(&mut self, input: &Blob) -> LayerForwardResult {
        Ok(vec![&self.output])
    }

    fn backward(
        &mut self,
        prev_input: Option<&Blob>,
        input: Option<&Blob>,
        weights: Option<&WsBlob>,
    ) -> LayerBackwardResult {
        Ok((&self.fake_lr.output, &self.fake_lr.ws))
    }

    fn learn_params(&mut self) -> Option<&mut LearnParams> {
        Some(&mut self.fake_lr)
    }

    fn layer_type(&self) -> &str {
        return "DummyLayer";
    }

    fn size(&self) -> usize {
        0
    }
}

impl DummyLayer {
    pub fn new() -> Self {
        DummyLayer {
            output: DataVec::zeros(0),
            fake_lr: LearnParams::new(0, 0),
        }
    }
}
