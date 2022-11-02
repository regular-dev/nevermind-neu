use super::util::DataVec;

pub fn sigmoid(val: f32) -> f32 {
    return 1.0 / (1.0 + (-val).exp());
}

pub fn sigmoid_deriv(val: f32) -> f32 {
    return (1.0 - val) * val;
}

pub fn tanh(val: f32) -> f32 {
    return val.tanh();
}

pub fn tanh_deriv(val: f32) -> f32 {
    return 1.0 - tanh(val) * tanh(val);
}

pub fn raw(val: f32) -> f32 {
    val
}

pub fn raw_deriv(val: f32) -> f32 {
    1.0
}

pub fn relu(val: f32) -> f32 {
    if val >= 0.0 {
        val
    } else {
        0.0
    }
}

pub fn relu_deriv(val: f32) -> f32 {
    1.0
}

pub fn sigmoid_on_vec(input: &DataVec, output: &mut DataVec) {
    if input.len() != output.len() {
        panic!("ERROR: input length vector != output length vector!!!");
    }

    for (idx, val) in input.indexed_iter() {
        output[idx] = sigmoid(*val);
    }
}

pub struct Activation<T: Fn(f32) -> f32, TD: Fn(f32) -> f32> {
    pub func: T,
    pub func_deriv: TD,
    pub name: String,
}

impl<T, TD> Activation<T, TD>
where
    T: Fn(f32) -> f32,
    TD: Fn(f32) -> f32,
{
    pub fn new(name: &str, func: T, func_deriv: TD) -> Self {
        Self {
            func,
            func_deriv,
            name: name.to_owned(),
        }
    }
}

pub mod activation_macros {
    #[macro_export]
    macro_rules! sigmoid_activation {
        (  ) => {{
            Activation::new("sigmoid", sigmoid, sigmoid_deriv)
        }};
    }

    #[macro_export]
    macro_rules! tanh_activation {
        (  ) => {{
            Activation::new("tanh", tanh, tanh_deriv)
        }};
    }

    #[macro_export]
    macro_rules! raw_activation {
        () => {
            Activation::new("raw", raw, raw_deriv)
        };
    }

    #[macro_export]
    macro_rules! relu_activation {
        () => {
            Activation::new("relu", relu, relu_deriv)
        };
    }

    pub use raw_activation;
    pub use relu_activation;
    pub use sigmoid_activation; // pub(crate) use sigmoid_activation
    pub use tanh_activation;
}
