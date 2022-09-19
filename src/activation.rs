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
    name: String, // &str ???
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
    macro_rules! sigmoid_activation {
        (  ) => {{
            Activation::new("sigmoid", sigmoid, sigmoid_deriv)
        }};
    }

    macro_rules! tanh_activation {
        (  ) => {{
            Activation::new("tanh", tanh, tanh_deriv)
        }};
    }

    pub(crate) use sigmoid_activation;
    pub(crate) use tanh_activation;
}

// pub trait ActivationTrait {
//     fn func(&self) -> Box<dyn Fn(f32) -> f32>;
//     fn name(&self) -> &str;
// }

// #[derive(Default)]
// pub struct SigmoidActivation;

// impl ActivationTrait for SigmoidActivation {
//     fn func(&self) -> Box<dyn Fn(f32) -> f32> {
//         Box::new(|val: f32| -> f32 { sigmoid(val) })
//     }

//     fn name(&self) -> &str {
//         "sigmoid"
//     }
// }

// #[derive(Default)]
// pub struct TanhActivation;

// impl ActivationTrait for TanhActivation {
//     fn func(&self) -> Box<dyn Fn(f32) -> f32> {
//         Box::new(|val: f32| -> f32 { tanh(val) })
//     }

//     fn name(&self) -> &str {
//         "tanh"
//     }
// }
