use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;

pub struct ErrorLayer {
    pub next_layer: Option< Box< dyn AbstractLayer > >,
    pub error: f32,
    pub expected_vals: Vec< f32 >
}

impl AbstractLayer for ErrorLayer {
    fn forward(&mut self, input: Blob) -> Blob
    {
        return Blob::new();
    }
    fn backward(&mut self, input: Blob) -> Blob
    {
        return Blob::new();
    }

    fn layer_name(&mut self) -> &str
    {
        "ErrorLayer"
    }

    fn next_layer(&mut self, _idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        if let Some(l) = &mut self.next_layer {
            return Some(l);
        } else {
            return None;
        }
    }
    fn previous_layer(&mut self, idx: usize) -> Option< &mut Box< dyn AbstractLayer > >
    {
        None
    }

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >)
    {

    }
}

impl ErrorLayer {
    pub fn new(size: i32) -> Self 
    {
        Self {
            next_layer: None,
            error: 0.0,
            expected_vals: Vec::new()
        }
    }
}