use std::io;
use std::error::Error;

fn read_from_stdin(stdin: &mut io::Stdin) -> Result<usize, Box<dyn Error>> {
    let mut inp_str = String::new();
    stdin.read_line(&mut inp_str)?;

    Ok(inp_str.parse::<usize>()?)
}

pub fn create_net() -> Result<(), Box<dyn Error>> {
    println!("Greetings traveller, tell me optimizator type : [sgd, rmsprop]");
    

    println!("Now tell me the input layer size");

    let mut inp_layer_str = String::new();
    let stdin = io::stdin(); 




    
    Ok(())
}