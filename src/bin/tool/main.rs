extern crate regular_mind;

use clap;
use clap::{App, Arg, ArgAction, Command};

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
                .help("Provides a file path to train dataset")
                .action(ArgAction::Set)
                .require_equals(true)
        )
        .arg(
            Arg::new("TestData")
                .long("test_dataset")
                .help("Provides a file path to test dataset")
                .action(ArgAction::Set)
                .require_equals(true)
        )
        .arg(
            Arg::new("SolverState")
                .short('s')
                .long("state")
                .help("Provide solver state. Weights state to start training")
                .action(ArgAction::Set)
                .require_equals(true)
        )
        .arg(
            Arg::new("SolverCfgYaml")
                .long("solver_cfg_yaml")
                .help("Provide solver configuration")
                .required(true)
                .action(ArgAction::Set)
                .require_equals(true)
        )
        .arg(
            Arg::new("MaxIter")
                .long("max_iter")
                .help("Provides maximum iteration number for training.")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
                .require_equals(true)
        )
        .arg(
            Arg::new("Err")
                .long("err")
                .help("Train till the net error will be less than this value")
                .action(ArgAction::Set).value_parser(clap::value_parser!(f32))
                .require_equals(true)
        )
        .arg(
            Arg::new("TestIter")
                .long("test_iter")
                .help("Each test_iter network will be tested for satisfying error")
                .action(ArgAction::Set).takes_value(true).value_parser(clap::value_parser!(usize))
                .require_equals(true)
        )
        .arg(
            Arg::new("SnapIter")
                .long("snap_iter")
                .help("Each snap_iter solver state will be saved")
                .action(ArgAction::Set).takes_value(true).value_parser(clap::value_parser!(usize))
                .require_equals(true)
        )
        .arg(Arg::new("TestBatch")
                .long("test_batch_size")
                .help("Provides test batch size")
                .action(ArgAction::Set)
                .takes_value(true)
                .value_parser(clap::value_parser!(usize))
                .require_equals(true)
        )
        .arg(Arg::new("WriteErrToFile").long("err_to_file").help(
            "Can be true or false, if true test network error will be recorded to file err.log",
        ).action(ArgAction::Set).require_equals(true)))
        .subcommand(Command::new("dataset_info").about("Inspect dataset"))
        .subcommand(Command::new("create_net").about("Create a new net configuration").arg(
            Arg::new("OutFile")
                .long("out")
                .short('o')
                .help("Specifies net configuration output file")
                .default_value("net.cfg")
                .action(ArgAction::Set))
            )
        .after_help("after help message. TODO : expand with examples")
        .get_matches();

    let cmd = matches.subcommand().unwrap();

    if cmd.0 == "dataset_info" {
        dataset_info::dataset_info(&matches)?;
    }
    if cmd.0 == "train" {
        let (_subcmd, args) = matches.subcommand().unwrap();
        train::train_new(&args)?;
    }
    if cmd.0 == "create_net" {
        let (_subcmd, args) = matches.subcommand().unwrap();
        create_net::create_net(&args)?;
    }

    Ok(())
}
