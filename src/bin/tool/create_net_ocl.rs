use std::error::Error;
use std::io;

use clap::ArgMatches;

use log::info;

use ocl::Queue;

use crate::create_net::*;
use nevermind_neu::err::*;
use nevermind_neu::layers::*;
use nevermind_neu::models::*;
use nevermind_neu::optimizers::*;
use nevermind_neu::util::*;

pub fn create_net_ocl(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let out_file = args.get_one::<String>("OutFile").unwrap();
    let out_optim = args.get_one::<String>("OptimFile").unwrap();

    let stdin = io::stdin();

    println!("Greetings traveller...");

    let seq_mdl = create_layers_ocl(&stdin)?;
    info!("Writing model configuration to file {}", out_file);
    seq_mdl.to_file(out_file)?;

    handle_optimizer_ocl(&stdin, &out_optim, seq_mdl.queue())?;

    Ok(())
}

fn handle_optimizer_ocl(
    stdin: &io::Stdin,
    filepath: &str,
    ocl_queue: Queue,
) -> Result<(), Box<dyn Error>> {
    println!("Would you like to create an optimizator configuration ? [y/n]");

    let yn: String = read_from_stdin(stdin)?;

    if yn == "n" || yn == "N" {
        return Ok(());
    }

    println!("Tell me optimizer type : [sgd, rmsprop, adagrad, adam]");
    let opt_type: String = read_from_stdin(&stdin)?;

    if opt_type == "sgd" {
        let mut optimizer = OptimizerOclSgd::new(ocl_queue);

        println!("Tell me learning rate [0.01 or 1e-2 format]");
        let lr: f32 = read_from_stdin(&stdin)?;

        println!("Tell me momentum [0.9 format]");
        let momentum: f32 = read_from_stdin(&stdin)?;

        optimizer.learn_rate = lr;
        optimizer.momentum = momentum;

        optimizer_ocl_to_file(optimizer, filepath)?;
    } else if opt_type == "rmsprop" {
        let mut optimizer = OptimizerOclRms::new(ocl_queue);

        println!("Tell me learning rate [0.01 or 1e-2 format]");
        let lr: f32 = read_from_stdin(&stdin)?;

        println!("Tell me alpha [0.9 format]");
        let alpha: f32 = read_from_stdin(&stdin)?;

        optimizer.learn_rate = lr;
        optimizer.alpha = alpha;

        optimizer_ocl_to_file(optimizer, filepath)?;
    } else if opt_type == "adagrad" {
        todo!()
    } else if opt_type == "adam" {
        todo!()
    }

    Ok(())
}

fn create_layers_ocl(stdin: &io::Stdin) -> Result<SequentialOcl, Box<dyn Error>> {
    let mut seq_mdl = SequentialOcl::new()?;

    // Input layer
    println!("Now tell me the input layer size");
    let inp_layer_size: usize = read_from_stdin(stdin)?;

    seq_mdl.add_layer(Box::new(InputLayerOcl::new(inp_layer_size)));

    // Hidden layer
    loop {
        println!("Do you want to add a hidden layer [y/n] ?");
        let mut answ: String = read_from_stdin(stdin)?;
        answ.make_ascii_lowercase();

        if answ == "y" {
            println!("Tell me the size of hidden layer");
            let l_size: usize = read_from_stdin(stdin)?;

            println!("Tell me the activation function for hidden layer [sigmoid/tanh/relu/leaky_relu/raw]");
            let mut answ: String = read_from_stdin(stdin)?;
            answ.make_ascii_lowercase();

            let act_res = OclActivationFunc::try_from(answ.as_str())?;
            seq_mdl.add_layer(Box::new(FcLayerOcl::new(l_size, act_res)));
        } else {
            break;
        }
    }

    println!("Tell me output layer activation function [raw, sigmoid, tanh, softmax_loss]");
    let out_l_type: String = read_from_stdin(stdin)?;

    println!("Tell me output layer size");
    let out_l_size: usize = read_from_stdin(stdin)?;

    if out_l_type == "softmax_loss" {
        todo!("Impl Softmax OCL");
    } else {
        let out_act = OclActivationFunc::try_from(out_l_type.as_str())?;
        seq_mdl.add_layer(Box::new(EuclideanLossLayerOcl::new_with_activation(
            out_l_size, out_act,
        )));
    }

    println!("Finally network : {}", seq_mdl);

    Ok(seq_mdl)
}
