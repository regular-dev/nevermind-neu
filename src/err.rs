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
            WrongArgErr => {
                write!(f, "{}", "Wrong arguments")
            },
            CustomError::Other => {
                write!(f, "{}", "Other")
            }
        }
    }
}

impl std::error::Error for CustomError {
    fn description(&self) -> &str {
        match self {
            WrongArgErr => {
                "Wrong arguments"
            },
            CustomError::Other => {
                "Other"
            }
        }
    }
}