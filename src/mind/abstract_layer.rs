use std::vec::Vec;

pub type DataVec = Vec< f32 >;
pub type Blob = Vec< DataVec >;

pub trait AbstractLayer {
    fn forward(&mut self, input: &Blob) -> &Blob;
    /// returns out_values and array of weights
    fn backward(&mut self, input: &Blob, weights: &Blob) -> (&Blob, &Blob);

    fn layer_name(&self) -> &str;

    fn next_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >;
    fn previous_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >;

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >);

    fn size(&self) -> usize;
}