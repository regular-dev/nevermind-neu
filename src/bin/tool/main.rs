extern crate nevermind_neu;

use clap;
use clap::{App, Arg, ArgAction, Command};

use env_logger::Env;

pub mod create_net;
pub mod dataset_info;
pub mod train;
pub mod test;
pub mod train_tui;

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
                .required(true)
        )
        .arg(
            Arg::new("TestData")
                .long("test_dataset")
                .help("Provides a file path to test dataset")
                .require_equals(true)
                .takes_value(true)
        )
        .arg(
            Arg::new("ModelState")
                .short('s')
                .long("state")
                .help("Provide model state. Weights state to start training")
                .takes_value(true)
                .require_equals(true)
        )
        .arg(
            Arg::new("ModelCfg")
                .long("model_cfg")
                .help("Provide model configuration")
                .required(true)
                .takes_value(true)
                .require_equals(true)
        )
        .arg(Arg::new("OptCfg")
                .short('o')
                .long("optimizer_cfg")
                .help("Provide optimizer configuration yaml file")
                .takes_value(true)
                .require_equals(true))
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
            Arg::new("Tui")
                .long("tui")
                .help("Use interactive terminal user interface")
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(bool))
                .require_equals(true)
                .default_value("false")
        )
        .arg(
            Arg::new("LrDecay")
                .long("lr_decay")
                .help("Sets the learning rate decay coefficient. Value between 0.0 - 1.0")
                .require_equals(true)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(f32))
        )
        .arg(
            Arg::new("LrStep")
                .long("lr_step")
                .help("Sets the step of learning rate decay")
                .require_equals(true)
                .action(ArgAction::Set)
                .value_parser(clap::value_parser!(usize))
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
                .help("Each snap_iter model state will be saved")
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
        .subcommand(Command::new("test").about("Test net").arg(
            Arg::new("Data")
            .long("dataset")
            .short('d')
            .takes_value(true)
            .require_equals(true)
            .required(true)
        )
        .arg(Arg::new("ModelCfg")
                .long("model_cfg")
                .help("Provide model configuration yaml file")
                .required(true)
                .takes_value(true)
                .require_equals(true))
        .arg(Arg::new("ModelState")
                .short('s')
                .long("state")
                .help("Provide state state. Weights state to continue training")
                .action(ArgAction::Set)
                .takes_value(true)
                .required(true)
                .require_equals(true))
        .arg(Arg::new("TestBatch")
                .long("test_batch")
                .short('b')
                .required(true)
                .require_equals(true)
                .takes_value(true)
                .value_parser(clap::value_parser!(usize))
        ))
        .subcommand(Command::new("dataset_info").about("Inspect dataset").arg(
            Arg::new("Data")
            .long("dataset")
            .short('d')
            .takes_value(true)
            .require_equals(true)
            
        ).arg( 
            Arg::new("ShowN")
            .long("show_n")
            .takes_value(true)
            .require_equals(true)
            .value_parser(clap::value_parser!(usize))
        ))
        .subcommand(Command::new("gen_init_state").about("Generate initial training state (with random weights)")
        .arg(
            Arg::new("ModelCfg")
            .long("model_cfg")
            .takes_value(true)
            .require_equals(true)
            .required(true)
        )
        .arg(
            Arg::new("OutFile")
            .long("out")
            .takes_value(true)
            .require_equals(true)
            .default_value("init.state")
        ))
        .subcommand(Command::new("create_net").about("Create a new net configuration").arg(
            Arg::new("OutFile")
                .long("out")
                .short('o')
                .help("Specifies model configuration output file")
                .default_value("net.cfg")
                .require_equals(true)
                .takes_value(true)
                .action(ArgAction::Set))
        .arg(Arg::new("OptimFile")
            .long("optim_out")
            .help("Specify optimizer configuration output file")
            .default_value("optim.cfg")))
        .after_help("after help message. TODO : expand with examples")
        .get_matches();

    let cmd = matches.subcommand().unwrap();

    if cmd.0 == "dataset_info" {
        let (_subcmd, args) = matches.subcommand().unwrap();
        dataset_info::dataset_info(&args)?;
    }
    if cmd.0 == "train" {
        let (_subcmd, args) = matches.subcommand().unwrap();

        let is_tui = args.get_one::<bool>("Tui").unwrap();

        if *is_tui {
            train_tui::train_tui(&args)?;
        } else {
            train::train_net(&args)?;
        }
    }
    if cmd.0 == "test" {
        let (_subcmd, args) = matches.subcommand().unwrap();
        test::test_net(&args)?;
    }
    if cmd.0 == "gen_init_state" {
        let (_subcmd, args) = matches.subcommand().unwrap();
        train::gen_init_state(&args)?;
    }
    if cmd.0 == "create_net" {
        let (_subcmd, args) = matches.subcommand().unwrap();
        create_net::create_net(&args)?;
    }

    Ok(())
}
