extern crate regular_mind;

use std::collections::HashMap;

use clap;
use clap::{App, Arg, ArgMatches, Command, ArgAction};

use env_logger::Env;

pub mod create_net;
pub mod dataset_info;
pub mod train;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let matches = App::new("regular_mind tool")
        .version("0.1.0")
        .author("Regular-dev")
        .about("Run neural network training, inspect datasets and more...")
        .subcommand_required(true)
        .subcommand(Command::new("train").about("Start a new training or continue").arg(
            Arg::new("TrainData")
                .long("train_dataset")
                .help("Provides a file path to train dataset"),
        )
        .arg(
            Arg::new("TestData")
                .long("test_dataset")
                .help("Provides a file path to test dataset"),
        )
        .arg(
            Arg::new("SolverState")
                .short('s')
                .long("state")
                .help("Provide solver state. Weights state to start training"),
        )
        .arg(
            Arg::new("SolverCfgYaml")
                .long("solver_cfg_yaml")
                .help("Provide solver configuration"),
        )
        .arg(
            Arg::new("MaxIter")
                .long("max_iter")
                .help("Provides maximum iteration number for training."),
        )
        .arg(
            Arg::new("Err")
                .long("err")
                .help("Train till the net error will be less than this value"),
        )
        .arg(
            Arg::new("TestIter")
                .long("test_iter")
                .help("Each test_iter network will be tested for satisfying error"),
        )
        .arg(
            Arg::new("SnapIter")
                .long("snap_iter")
                .help("Each snap_iter solver state will be saved"),
        )
        .arg(Arg::new("WriteErrToFile").long("err_to_file").help(
            "Can be true or false, if true test network error will be recorded to file err.log",
        )))
        .subcommand(Command::new("dataset_info").about("Inspect dataset"))
        .subcommand(Command::new("create_net").about("Create a new net configuration").arg
        (Arg::new("OutFile").long("out").short('o').help("Specifies net configuration output file").required(true).action(ArgAction::Set)))
        .after_help("after help message. TODO : expand with examples")
        .get_matches();

    let cmd = matches.subcommand().unwrap();

    if cmd.0 == "dataset_info" {
        dataset_info::dataset_info(&matches)?;
    }
    if cmd.0 == "train" {
        train::train_new(&matches)?;
    }
    if cmd.0 == "create_net" {
        create_net::create_net()?;
    }

    Ok(())
}
