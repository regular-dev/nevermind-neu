use log::debug;

use crate::util::{DataVec, Num, WsMat};

pub trait Bias {
    fn forward(&mut self, ws: &WsMat) -> &DataVec;
}

#[derive(Default)]
pub struct ConstBias {
    pub val: Num,
    pub output: DataVec,
}

impl ConstBias {
    pub fn new(size: usize, val: Num) -> Self {
        Self {
            val,
            output: DataVec::zeros(size),
        }
    }

    // TODO : impl sized ConstBias -> [1.0, 1.0, 1.0] with weights.
}

impl Bias for ConstBias {
    fn forward(&mut self, ws: &WsMat) -> &DataVec {
        self.output = (self.val * ws).into_shape(self.output.len()).unwrap();
        &self.output
    }
}
