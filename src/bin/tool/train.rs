use log::{error, info};

use std::time::Instant;

use serde::Serialize;

use clap::ArgMatches;

// regular_mind
use regular_mind::dataloader::*;
use regular_mind::network::*;
use regular_mind::solvers::*;
use regular_mind::err::*;

/// Starts train a network with required net configuration
/// and train dataset
pub fn train_new(
    args: &ArgMatches
) -> Result<(), Box<dyn std::error::Error>> {
    if !args.contains_id("SolverCfgYaml") {
        error!("Solver configuration wasn't provided (--solver_cfg_yaml)");
        return Err(Box::new(CustomError::WrongArg));
    }

    let solver_cfg = args.get_one::<String>("SolverCfgYaml").unwrap();
    let solver_type = solver_type_from_file(solver_cfg)?;

    match solver_type.as_str() {
        "rmsprop" => {
            let solver = SolverRMS::from_file(solver_cfg)?;
            train_net(solver, args)
        },
        "sgd" => {
            let solver = SolverSGD::from_file(solver_cfg)?;
            train_net(solver, args)
        }
        _ => {
            Err(Box::new(CustomError::Other))
        }
    }
}

pub fn train_net(
    mut solver: impl Solver + Serialize,
    args: &ArgMatches
) -> Result<(), Box<dyn std::error::Error>> {
    if !args.contains_id("TrainData") {
        error!("TrainData wasn't provided (--train_dataset)");
        return Err(Box::new(CustomError::WrongArg));
    }

    let train_ds = args.get_one::<String>("TrainData").unwrap();
    let train_ds = Box::new(ProtobufDataLoader::from_file(train_ds)?);

    if let Some(state_file) = args.get_one::<String>("SolverState") {
        info!("Loading solver state from file {}", *state_file);
        solver.load_state(state_file)?;
    }

    let mut net = Network::new(train_ds, solver);

    // Set test data if exists argument
    if let Some(test_ds) = args.get_one::<String>("TestData") {
        info!("Setting test data : {}", test_ds);
        let test_ds = Box::new(ProtobufDataLoader::from_file(test_ds)?);
        net = net.test_dataloader(test_ds);
    }

    if let Some(test_iter) = args.get_one::<usize>("TestIter") {
        info!("Test iter : {}", test_iter);
        net = net.test_iter(*test_iter as usize);
    }

    if let Some(snap_iter) = args.get_one::<usize>("SnapIter") {
        info!("Snapshot iter : {}", snap_iter);
        net = net.snap_iter(*snap_iter);
    }

    if let Some(test_batch) = args.get_one::<usize>("TestBatch") {
        info!("Test batch size : {}", test_batch);
        net = net.test_batch_num(*test_batch);
    }

    if let Some(write_test_err) = args.get_one::<String>("WriteErrToFile") {
        let is_true = write_test_err.eq("true") || write_test_err.eq("TRUE");

        if is_true {
            info!("Writing error is enabled");
        } else {
            info!("Writing error is disabled");
        }

        net = net.write_test_err_to_file(is_true);
    }

    let mut opt_err = None;
    let mut opt_max_iter = None;
    
    if let Some(err) = args.get_one::<f32>("Err") {
        info!("Satisfying error : {}", err);
        opt_err = Some(err);
    }

    if let Some(max_iter) = args.get_one::<usize>("MaxIter") {
        info!("Satisfying error : {}", max_iter);
        opt_max_iter = Some(max_iter);
    }

    let now_time = Instant::now();

    if opt_err.is_some() && opt_max_iter.is_some() {
        let err = opt_err.unwrap();
        let max_iter = opt_max_iter.unwrap();

        info!("Start train till the err {} or max iteration {}", *err, *max_iter);
        net.train_for_error_or_iter(*err, *max_iter)?;
    } else if opt_err.is_some() {
        let err = opt_err.unwrap();

        info!("Start train till the err {}", *err);
        net.train_for_error(*err)?;
    } else if opt_max_iter.is_some() {
        let max_iter = opt_max_iter.unwrap();

        info!("Start train max iteration {}", *max_iter);
        net.train_for_n_times(*max_iter)?;
    } else {
        error!("Error and max iteration for training wasn't set (--max_iter , -err)");
        return Err(Box::new(CustomError::WrongArg));
    }

    let elapsed_bench = now_time.elapsed();
    
    info!("Elapsed for training : {} ms", elapsed_bench.as_millis());

    Ok(())
}
