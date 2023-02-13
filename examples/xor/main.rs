use std::fs::File;

use log::{info, LevelFilter, SetLoggerError};

use std::time::Instant;

use log4rs::config::{Appender, Config, Root};

use ndarray::array;

use env_logger::Env;

use nevermind_neu::dataloader::*;
use nevermind_neu::network::*;
use nevermind_neu::models::*;
use nevermind_neu::optimizers::*;

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
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();
    log::info!("Regular-mind 0.1 test app starting...");

    // prepare data set
    let mut dataset_train: Vec<DataBatch> = Vec::new();
    dataset_train.push(DataBatch::new(vec![0.0, 0.0], vec![0.0]));
    dataset_train.push(DataBatch::new(vec![0.0, 1.0], vec![1.0]));
    dataset_train.push(DataBatch::new(vec![1.0, 0.0], vec![1.0]));
    dataset_train.push(DataBatch::new(vec![1.0, 1.0], vec![0.0]));

    let dataloader = Box::new(SimpleDataLoader::new(dataset_train));

    //let mut solver_rms = SolverRMS::from_file("network.cfg")?;
    // solver_rms.load_state("solver_state.proto")?;

    let net_cfg = vec![2, 10, 1];
    let mut seq_mdl = Sequential::new_simple(&net_cfg);
    seq_mdl.set_batch_size(4);

    let opt = Box::new(OptimizerRMS::new(1e-2, 0.8));

    let mut net = Network::new(seq_mdl).test_batch_num(4);

    net.set_optimizer(opt);
    net.set_train_dataset(dataloader);

    //  net.save_network_cfg("network.cfg")?;

    let now_time = Instant::now();

    net.train_for_error_or_iter(0.05, 2000)?;

    let elapsed_bench = now_time.elapsed();

    info!("Elapsed for training : {} ms", elapsed_bench.as_millis());

    //net.save_solver_state("solver_state.proto")?;

    // test dataset
    let mut dataset_test: Vec<DataBatch> = Vec::new();
    dataset_test.push(DataBatch::new(vec![0.0, 0.0], vec![0.0]));
    dataset_test.push(DataBatch::new(vec![0.0, 1.0], vec![1.0]));
    dataset_test.push(DataBatch::new(vec![1.0, 0.0], vec![1.0]));
    dataset_test.push(DataBatch::new(vec![1.0, 1.0], vec![0.0]));

    info!("Now testing net !!!");

    let out = net.eval(array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).unwrap();
    let out_b = out.borrow();

    info!("Trained-net XOR out : {}", out_b);

    Ok(())
}
