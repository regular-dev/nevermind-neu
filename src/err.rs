use std::fmt;

#[derive(Debug)]
pub enum CustomError {
    WrongArg,
    InvalidFormat,
    Other
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CustomError::WrongArg => {
                write!(f, "{}", "Wrong arguments")
            },
            _ => {
                write!(f, "{}", "Other")
            },
        }
    }
}

impl std::error::Error for CustomError {
    fn description(&self) -> &str {
        match self {
            CustomError::WrongArg => {
                "Wrong arguments"
            },
            _ => {
                "Other"
            }
        }
    }
}