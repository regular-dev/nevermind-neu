use crate::mind::abstract_layer::AbstractLayer;
use crate::mind::abstract_layer::Blob;

pub struct HiddenLayer {
    pub ws: Vec< f32 >,
    pub next_layer: Option< Box< dyn AbstractLayer > >,
    pub prev_layer: Option< Box< dyn AbstractLayer > >
}

impl AbstractLayer for HiddenLayer {
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
        "HiddenLayer"
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
        if let Some(l) = &mut self.prev_layer {
            return Some(l);
        } else {
            return None;
        }
    }

    fn add_next_layer(&mut self, layer: Box< dyn AbstractLayer >)
    {
        self.next_layer = Some(layer);
    }
}

impl HiddenLayer {
    pub fn new(size: i32) -> Self {
        Self {
            ws: Vec::new(),
            next_layer: None,
            prev_layer: None,
        }
    }
}

