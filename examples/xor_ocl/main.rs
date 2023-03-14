use log::info;

use std::time::Instant;

use env_logger::Env;

use ndarray_rand::rand::thread_rng;

use nevermind_neu::dataloader::*;
use nevermind_neu::network::*;
use nevermind_neu::models::*;
use nevermind_neu::optimizers::*;

#[cfg(feature = "log_env_logger")]
fn init_logger() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();
    log::info!("nevermind-neu xor_ocl example starting...");

    // prepare data set
    let mut dataset_train: Vec<LabeledEntry> = Vec::new();
    dataset_train.push(LabeledEntry::new(vec![0.0, 0.0], vec![0.0]));
    dataset_train.push(LabeledEntry::new(vec![0.0, 1.0], vec![1.0]));
    dataset_train.push(LabeledEntry::new(vec![1.0, 0.0], vec![1.0]));
    dataset_train.push(LabeledEntry::new(vec![1.0, 1.0], vec![0.0]));

    let mut dataloader = Box::new(SimpleDataLoader::new(dataset_train));

    //let mut solver_rms = SolverRMS::from_file("network.cfg")?;
    // solver_rms.load_state("solver_state.proto")?;

    let net_cfg = vec![2, 10, 1];
    let mut seq_mdl = SequentialOcl::new_simple(&net_cfg);
    seq_mdl.set_batch_size(4);

    let mut opt = Box::new(OptimizerOclRms::new(seq_mdl.queue()));

    //let mut net = Network::new(seq_mdl).test_batch_num(4);

    //net.set_optimizer(opt);
    //net.set_train_dataset(dataloader);

    //  net.save_network_cfg("network.cfg")?;
    for _ in 0..1500 {
      let t_data = dataloader.next_batch(4);
      seq_mdl.feedforward(t_data.input);
      seq_mdl.backpropagate(t_data.output);
      seq_mdl.optimize(&mut opt);
    }

    return Ok(());

    let now_time = Instant::now();

  //  net.train_for_error_or_iter(0.05, 2000)?;

    let elapsed_bench = now_time.elapsed();

    info!("Elapsed for training : {} ms", elapsed_bench.as_millis());

    //net.save_solver_state("solver_state.proto")?;

    info!("Now testing net !!!");

  //  let out = net.eval(array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).unwrap();
   // let out_b = out.borrow();

  //  info!("Trained-net XOR out : {}", out_b);

    Ok(())
}
