use ndarray::{Array, Axis};

use crate::util::{DataVec, Batch};


#[derive(Clone, Default)]
pub struct DataBatch {
    pub input: DataVec,
    pub expected: DataVec,
}

impl DataBatch {
    pub fn new(input: Vec<f32>, expected: Vec<f32>) -> Self {
        Self {
            input: Array::from_vec(input),
            expected: Array::from_vec(expected),
        }
    }
}

#[derive(Default, Clone)]
pub struct MiniBatch {
    pub input: Batch,
    pub output: Batch,
}

impl MiniBatch {
    pub fn new(b: Vec<&DataBatch>) -> Self {
        assert!( !b.is_empty() );

        let mut inp_arr = Batch::zeros( (b.len(), b.first().unwrap().input.shape()[0]) );
        let mut out_arr = Batch::zeros( (b.len(), b.first().unwrap().expected.shape()[0]) );

        // Copies memory into batch

        for (idx, it) in b.iter().enumerate() {
            let mut inp_entry = inp_arr.index_axis_mut(Axis(0), idx);
            inp_entry.assign(&it.input);

            let mut out_entry = out_arr.index_axis_mut(Axis(0), idx);
            out_entry.assign(&it.expected);
        }

        Self {
            input: inp_arr,
            output: out_arr,
        }
    }
}