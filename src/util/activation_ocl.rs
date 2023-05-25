use std::fmt::{self, Debug};

#[derive(Clone, Debug)]
pub enum OclActivationFunc {
    Raw,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
}

impl fmt::Display for OclActivationFunc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let act_str = match self {
            OclActivationFunc::Raw => "raw",
            OclActivationFunc::Sigmoid => "sigmoid",
            OclActivationFunc::Tanh => "tanh",
            OclActivationFunc::ReLU => "relu",
            OclActivationFunc::LeakyReLU => "leaky_relu",
        };

        write!(f, "{}", act_str)
    }
}

impl TryFrom<&str> for OclActivationFunc {
    type Error = &'static str;

    fn try_from(input: &str) -> Result<Self, Self::Error> {
        match input {
            "sigmoid" => Ok(OclActivationFunc::Sigmoid),
            "tanh" => Ok(OclActivationFunc::Tanh),
            "relu" => Ok(OclActivationFunc::ReLU),
            "leaky_relu" => Ok(OclActivationFunc::LeakyReLU),
            "raw" => Ok(OclActivationFunc::Raw),
            _ => Err("Invalid input string"),
        }
    }
}

pub static OCL_ACTIVATION_SIGMOID: &'static str = r#"
    float activation(float v)
    {
        return 1.0 / (1.0 + pow(M_E_F, -v));
    }
"#;

pub static OCL_ACTIVATION_SIGMOID_DERIV: &'static str = r#"
    float sigmoid(float v)
    {
        return 1.0 / (1.0 + pow(M_E_F, -v));
    }

    float deriv(float v) 
    {
        return (1.0 - sigmoid(v)) * sigmoid(v);
    }
"#;

pub static OCL_ACTIVATION_TANH: &'static str = r#"
    float activation(float v)
    {
        return tanh( v );
    }
"#;

pub static OCL_ACTIVATION_TANH_DERIV: &'static str = r#"
    float deriv(float v)
    {
        return 1.0 - pow(tanh(v), 2);
    }
"#;

pub static OCL_ACTIVATION_RELU: &'static str = r#"
    float activation(float v)
    {
        return v > 0.0 ? v : 0.0;
    }
"#;

pub static OCL_ACTIVATION_RELU_DERIV: &'static str = r#"
    float deriv(float v)
    {
        return v > 0.0 ? 1.0 : 0.0;
    }
"#;

pub static OCL_ACTIVATION_LEAKY_RELU: &'static str = r#"
    float activation(float v)
    {
        return v > 0.0 ? v : 0.01 * v;
    }
"#;

pub static OCL_ACTIVATION_LEAKY_RELU_DERIV: &'static str = r#"
    float deriv(float v)
    {
        return v > 0.0 ? 1.0 : 0.01;
    }
"#;

pub static OCL_ACTIVATION_RAW: &'static str = r#"
    float activation(float v)
    {
        return v;
    }
"#;

pub static OCL_ACTIVATION_RAW_DERIV: &'static str = r#"
    float deriv(float v)
    {
        return 1.0;
    }
"#;
