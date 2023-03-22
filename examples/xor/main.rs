use log::info;

use std::time::Instant;

use ndarray::array;
use env_logger::Env;

use nevermind_neu::dataloader::*;
use nevermind_neu::orchestra::*;
use nevermind_neu::models::*;
use nevermind_neu::optimizers::*;

#[cfg(feature = "log_env_logger")]
fn init_logger() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();
    log::info!("nevermind-neu xor example starting...");

    // prepare data set
    let mut dataset_train: Vec<LabeledEntry> = Vec::new();
    dataset_train.push(LabeledEntry::new(vec![0.0, 0.0], vec![0.0]));
    dataset_train.push(LabeledEntry::new(vec![0.0, 1.0], vec![1.0]));
    dataset_train.push(LabeledEntry::new(vec![1.0, 0.0], vec![1.0]));
    dataset_train.push(LabeledEntry::new(vec![1.0, 1.0], vec![0.0]));

    let dataloader = Box::new(SimpleDataLoader::new(dataset_train));

    let net_cfg = vec![2, 10, 1];
    let mut seq_mdl = Sequential::new_simple(&net_cfg);
    seq_mdl.set_batch_size(4);

    let opt = Box::new(OptimizerRMS::new(1e-2, 0.8));
    seq_mdl.set_optim(opt);

    let mut net = Orchestra::new(seq_mdl).test_batch_size(4);

    net.set_save_on_finish_flag(false);
    net.set_train_dataset(dataloader);

    let now_time = Instant::now();

    net.train_for_error_or_iter(0.05, 2000)?;

    let elapsed_bench = now_time.elapsed();

    info!("Elapsed for training : {} ms", elapsed_bench.as_millis());

    info!("Now testing net !!!");

    let out = net.eval(array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).unwrap();
    let out_b = out.borrow();

    info!("Trained-net XOR out : {}", out_b);

    Ok(())
}
