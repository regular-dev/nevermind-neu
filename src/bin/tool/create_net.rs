use std::error::Error;
use std::io;
use std::str::FromStr;

use serde::Serialize;

use clap::ArgMatches;

use log::info;

use regular_mind::err::*;
use regular_mind::layers_storage::*;
use regular_mind::solvers::*;
use regular_mind::layers::*;
use regular_mind::network::*;
use regular_mind::activation::*;

fn read_from_stdin<T: FromStr>(stdin: &io::Stdin) -> Result<T, Box<dyn Error>> {
    let mut inp_str = String::new();
    stdin.read_line(&mut inp_str)?;
    inp_str.pop(); // TODO : fint better way to crop \0 ?

    let out = inp_str.parse::<T>();

    if let Ok(out) = out {
        return Ok(out);
    }

    return Err(Box::new(CustomError::Other));
}

pub fn create_net(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let out_file = args.get_one::<String>("OutFile").unwrap();

    let stdin = io::stdin();

    println!("Greetings traveller, tell me solver type : [sgd, rmsprop]");
    let solver_type: String = read_from_stdin(&stdin)?;

    if solver_type.as_str() == "sgd" {
        let solver = create_net_sgd(&stdin)?;

        info!("Writing net configuration to file : {}", out_file);
        save_solver_cfg(&solver, out_file)?;
    } else if solver_type.as_str() == "rmsprop" {
        let solver = create_net_rms(&stdin)?;

        info!("Writing net configuration to file : {}", out_file);
        save_solver_cfg(&solver, out_file)?;
    } else {
        return Err(Box::new(CustomError::Other));
    }

    Ok(())
}

fn create_net_rms(stdin: &io::Stdin) -> Result<SolverRMS, Box<dyn Error>> {
    let solver = SolverRMS::new();
    let solver = create_layers(stdin, solver);

    solver
}

fn create_net_sgd(stdin: &io::Stdin) -> Result<SolverSGD, Box<dyn Error>> {
    let solver = SolverSGD::new();
    let solver = create_layers(stdin, solver);

    solver
}

fn create_layers<T: Solver + Serialize>(
    stdin: &io::Stdin,
    mut solver: T,
) -> Result<T, Box<dyn Error>> {
    let mut ls = LayersStorage::new();

    // Input layer
    println!("Now tell me the input layer size");
    let inp_layer_size: usize = read_from_stdin(stdin)?;
    ls.add_layer(Box::new(InputDataLayer::new(inp_layer_size)));

    // Hidden layer
    let mut prev_s = inp_layer_size;
    loop {
        println!("Do you want to add a hidden layer [y/n] ?");
        let mut answ: String = read_from_stdin(stdin)?;
        answ.make_ascii_lowercase();

        if answ == "y" {
            println!("Tell me the size of hidden layer");
            let l_size: usize = read_from_stdin(stdin)?;

            println!("Tell me the activation function for hidden layer [sigmoid/tanh/relu/raw]");
            let mut answ: String = read_from_stdin(stdin)?;
            answ.make_ascii_lowercase();

            match answ.as_str() {
                "sigmoid" => {
                    ls.add_layer(Box::new(HiddenLayer::new(l_size, prev_s, activation_macros::sigmoid_activation!())));
                },
                "tanh" => {
                    ls.add_layer(Box::new(HiddenLayer::new(l_size, prev_s, activation_macros::tanh_activation!())));
                },
                "relu" => {
                    ls.add_layer(Box::new(HiddenLayer::new(l_size, prev_s, activation_macros::relu_activation!())));
                },
                "raw" => {
                    ls.add_layer(Box::new(HiddenLayer::new(l_size, prev_s, activation_macros::raw_activation!())));
                },
                _ => {
                    return Err(Box::new(CustomError::WrongArg));
                }
            }
            prev_s = l_size;
        } else {
            break;
        }
    }

    println!("Tell me output layer size");
    let out_l_size:usize = read_from_stdin(stdin)?;

    ls.add_layer(Box::new(HiddenLayer::new(
        out_l_size,
        prev_s,
        activation_macros::raw_activation!(),
    )));

    println!("Finally network : {}", ls);

    solver.setup_network(ls);

    Ok(solver)
}
