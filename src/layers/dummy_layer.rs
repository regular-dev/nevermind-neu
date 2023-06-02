use std::option::Option;

use crate::cpu_params::{CpuParams, ParamsBlob};
use crate::util::{DataVec, WithParams};
use crate::layers::{AbstractLayer, LayerBackwardResult, LayerForwardResult};

// not used
#[derive(Default, Clone)]
pub struct DummyLayer {
    output: DataVec,
    fake_lr: CpuParams,
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

    fn cpu_params(&self) -> Option<CpuParams> {
        Some(self.fake_lr.clone())
    }

    fn layer_type(&self) -> &str {
        return "DummyLayer";
    }

    fn set_input_shape(&mut self, sh: &[usize]) {
    }

    fn size(&self) -> usize {
        0
    }

    fn copy_layer(&self) -> Box<dyn AbstractLayer> {
        Box::new(self.clone())
    }

    fn clone_layer(&self) -> Box<dyn AbstractLayer> {
        self.copy_layer()
    }

    fn set_cpu_params(&mut self, lp: CpuParams) {
        self.fake_lr = lp;
    }
}

impl DummyLayer {
    pub fn new() -> Self {
        DummyLayer {
            output: DataVec::zeros(0),
            fake_lr: CpuParams::new(0, 0),
        }
    }
}

impl WithParams for DummyLayer { }
