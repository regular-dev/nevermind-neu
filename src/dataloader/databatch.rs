use ndarray::{Array, Axis};

use crate::util::{DataVec, Array2D};


#[derive(Clone, Default)]
pub struct LabeledEntry {
    pub input: DataVec,
    pub expected: DataVec,
}

impl LabeledEntry {
    pub fn new(input: Vec<f32>, expected: Vec<f32>) -> Self {
        Self {
            input: Array::from_vec(input),
            expected: Array::from_vec(expected),
        }
    }
}

#[derive(Default, Clone)]
pub struct MiniBatch {
    pub input: Array2D,
    pub output: Array2D,
}

impl MiniBatch {
    pub fn new(b: Vec<&LabeledEntry>) -> Self {
        assert!( !b.is_empty() );

        let mut inp_arr = Array2D::zeros( (b.len(), b.first().unwrap().input.shape()[0]) );
        let mut out_arr = Array2D::zeros( (b.len(), b.first().unwrap().expected.shape()[0]) );

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

    pub fn new_no_ref(b: Vec<LabeledEntry>) -> Self {
        // TODO : refactor cause dublicating constructors
        assert!( !b.is_empty() );

        let mut inp_arr = Array2D::zeros( (b.len(), b.first().unwrap().input.shape()[0]) );
        let mut out_arr = Array2D::zeros( (b.len(), b.first().unwrap().expected.shape()[0]) );

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