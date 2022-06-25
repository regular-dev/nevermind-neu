use std::vec::Vec;

pub type DataVec = Vec< f32 >;
pub type Blob = Vec< DataVec >;

pub trait AbstractLayer {
    fn forward(&mut self, input: &Blob) -> &Blob;
    /// returns out_values and array of weights
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob);
    fn optimize(&mut self, prev_out: &Blob) -> &Blob;

    fn layer_name(&self) -> &str;

    fn size(&self) -> usize;
}