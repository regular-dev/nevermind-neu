use std::error::Error;
use std::io;
use std::str::FromStr;

use serde::Serialize;

use clap::ArgMatches;

use log::info;

use regular_mind::activation::*;
use regular_mind::err::*;
use regular_mind::layers::*;
use regular_mind::layers_storage::*;
use regular_mind::network::*;
use regular_mind::models::*;
use regular_mind::optimizers::*;


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
    let out_optim = args.get_one::<String>("OptimFile").unwrap();

    let stdin = io::stdin();

    println!("Greetings traveller...");

    let seq_mdl = create_layers(&stdin)?;
    info!("Writing model configuration to file {}", out_file);
    seq_mdl.to_file(out_file)?;

    handle_optimizer(&stdin, &out_optim)?;

    Ok(())
}

fn handle_optimizer(stdin: &io::Stdin, filepath: &str) -> Result<(), Box<dyn Error>> {
    println!("Would you like to create an optimizator configuration ? [y/n]");

    let yn: String = read_from_stdin(stdin)?;

    if yn == "n" || yn == "N" {
        return Ok(());
    }

    println!("Tell me optimizer type : [sgd, rmsprop]");
    let opt_type: String = read_from_stdin(&stdin)?;

    if opt_type == "sgd" {
        let mut optimizer = OptimizerSGD::default();

        println!("Tell me learning rate [0.01 format]");
        let lr: f32 = read_from_stdin(&stdin)?;

        println!("Tell me momentum [0.8 format]");
        let momentum: f32 = read_from_stdin(&stdin)?;

        optimizer.learn_rate = lr;
        optimizer.momentum = momentum;

        optimizer_to_file(optimizer, filepath)?;
    } else if opt_type == "rmsprop" {
        let mut optimizer = OptimizerRMS::default();

        println!("Tell me learning rate [0.01 format]");
        let lr: f32 = read_from_stdin(&stdin)?;

        println!("Tell me alpha [0.8 format]");
        let alpha: f32 = read_from_stdin(&stdin)?;

        optimizer.learn_rate = lr;
        optimizer.alpha = alpha;

        optimizer_to_file(optimizer, filepath)?;
    }

    Ok(())
}

fn create_layers(
    stdin: &io::Stdin,
) -> Result<Sequential, Box<dyn Error>> {
    let mut ls = SequentialLayersStorage::empty();

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
                    ls.add_layer(Box::new(HiddenLayer::new(
                        l_size,
                        prev_s,
                        activation_macros::sigmoid_activation!(),
                    )));
                }
                "tanh" => {
                    ls.add_layer(Box::new(HiddenLayer::new(
                        l_size,
                        prev_s,
                        activation_macros::tanh_activation!(),
                    )));
                }
                "relu" => {
                    ls.add_layer(Box::new(HiddenLayer::new(
                        l_size,
                        prev_s,
                        activation_macros::relu_activation!(),
                    )));
                }
                "raw" => {
                    ls.add_layer(Box::new(HiddenLayer::new(
                        l_size,
                        prev_s,
                        activation_macros::raw_activation!(),
                    )));
                }
                _ => {
                    return Err(Box::new(CustomError::WrongArg));
                }
            }
            prev_s = l_size;
        } else {
            break;
        }
    }

    println!("Tell me output layer type [raw, softmax_loss]");
    let out_l_type: String = read_from_stdin(stdin)?;

    println!("Tell me output layer size");
    let out_l_size: usize = read_from_stdin(stdin)?;

    if out_l_type == "raw" {
        ls.add_layer(Box::new(ErrorLayer::new(
            out_l_size,
            prev_s,
            activation_macros::raw_activation!(),
        )));
    } else if out_l_type == "softmax_loss" {
     //   ls.add_layer(Box::new(SoftmaxLossLayer::new(out_l_size, prev_s)));
    }

    println!("Finally network : {}", ls);

    let seq_mdl = Sequential::new_with_layers(ls);

    Ok(seq_mdl)
}
