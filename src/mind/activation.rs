use super::abstract_layer::DataVec;


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

    for (idx, val) in input.iter().enumerate() {
        output[idx] = sigmoid(*val);
    }
}