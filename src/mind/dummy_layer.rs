use std::option::Option;

use std::cell::RefCell;
use std::rc::Rc;

use super::learn_params::{LearnParams, LearnParamsPtr, ParamsBlob};
use super::util::{Blob, DataVec, Variant, WsBlob, WsMat};
use crate::mind::abstract_layer::{AbstractLayer, LayerBackwardResult, LayerForwardResult};

// not used
pub struct DummyLayer {
    output: DataVec,
    fake_lr: LearnParams,
}

impl AbstractLayer for DummyLayer {
    fn forward(&mut self, input: ParamsBlob) -> LayerForwardResult {
        Ok(input)
    }

    fn backward(
        &mut self,
        prev_input: ParamsBlob,
        input: ParamsBlob,
    ) -> LayerBackwardResult {
        Ok( vec![self.fake_lr.clone()] )
    }

    fn learn_params(&self) -> Option<LearnParams> {
        Some(self.fake_lr.clone())
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
