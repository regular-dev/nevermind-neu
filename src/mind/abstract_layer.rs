use std::vec::Vec;


pub type Blob = Vec< Vec <f32> >;

pub trait AbstractLayer {
    fn forward(&mut self, input: Blob) -> Blob;
    fn backward(&mut self, input: Blob) -> Blob;

    fn layer_name(&mut self) -> &str;

    fn next_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >;
    fn previous_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >;

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >);
}