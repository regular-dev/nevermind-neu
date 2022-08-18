mod mind;

use log::{LevelFilter, SetLoggerError, info};
use log4rs::append::console::ConsoleAppender;
use log4rs::append::file::FileAppender;
use log4rs::encode::pattern::PatternEncoder;

use log4rs::config::{Appender, Config, Root};

use env_logger::Env;

use crate::mind::dataset::DataBatch;
use crate::mind::dataset::SimpleDataLoader;
use crate::mind::network::Network;
use crate::mind::solver_sgd::SolverSGD;
use crate::mind::solver_rmsprop::SolverRMS;


#[cfg(feature = "log_log4rs")]
fn init_logger() {
    // env_logger::init();
    // simple_logging::log_to_file("log.txt", LevelFilter::Warn);

    let logfile_res = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::default()))
        .build("log.txt");

    let console_res = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::default()))
        .build();

    if let Ok(logfile) = logfile_res {
        let config = Config::builder()
            .appender(Appender::builder().build("logfile", Box::new(logfile)))
            .appender(Appender::builder().build("console", Box::new(console_res)))
            .build(
                Root::builder()
                    .appender("console")
                    .appender("logfile")
                    .build(LevelFilter::Info),
            )
            .unwrap();

        let init_res = log4rs::init_config(config);

        match init_res {
            Ok(_val) => {
                return;
            }
            Err(err) => {
                panic!("Couldn't initialize logger !!!");
            }
        }
    } else {
        panic!("Couldn't initialize logger !!!");

    }
}

#[cfg(feature = "log_env_logger")]
fn init_logger() {
    env_logger::Builder::from_env(Env::default().default_filter_or("warn")).init();
}

fn main() {
    init_logger();
    log::info!("Regular-mind 0.1 test app starting...");

    // prepare data set
    let mut dataset_train: Vec<DataBatch> = Vec::new();
    dataset_train.push(DataBatch::new(vec![0.0, 0.0], vec![0.0]));
    dataset_train.push(DataBatch::new(vec![0.0, 1.0], vec![1.0]));
    dataset_train.push(DataBatch::new(vec![1.0, 0.0], vec![1.0]));
    dataset_train.push(DataBatch::new(vec![1.0, 1.0], vec![0.0]));

    let dataloader = Box::new(SimpleDataLoader::new(dataset_train));

    // create a network
    let mut net = Network::new(dataloader, SolverRMS::new().batch(4));
    let net_cfg = vec![2, 10, 1];
    net.setup_simple_network(&net_cfg);

    net.save_network_cfg("network.cfg");
    net.train_for_n_times(30000);

    // test dataset
    let mut dataset_test: Vec<DataBatch> = Vec::new();
    dataset_test.push(DataBatch::new(vec![0.0, 0.0], vec![0.0]));
    dataset_test.push(DataBatch::new(vec![0.0, 1.0], vec![1.0]));
    dataset_test.push(DataBatch::new(vec![1.0, 0.0], vec![1.0]));
    dataset_test.push(DataBatch::new(vec![1.0, 1.0], vec![0.0]));

    info!("Now testing net !!!");

    net.feedforward(&dataset_test[0], true);
    net.feedforward(&dataset_test[1], true);
    net.feedforward(&dataset_test[2], true);
    net.feedforward(&dataset_test[3], true);
}
