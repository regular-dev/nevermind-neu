use log::{error, info};

use std::time::Instant;

use serde::Serialize;

use clap::ArgMatches;

// regular_mind
use regular_mind::dataloader::*;
use regular_mind::err::*;
use regular_mind::network::*;
use regular_mind::solvers::*;

pub fn test_net(args: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let solver_cfg = args.get_one::<String>("SolverCfgYaml").unwrap();
    let solver_state = args.get_one::<String>("SolverState").unwrap();
    let solver_type = solver_type_from_file(solver_cfg)?;

    match solver_type.as_str() {
        "rmsprop" => {
            let mut solver = SolverRMS::from_file(solver_cfg)?;
            solver.load_state(solver_state)?;
            test_net_helper(args, solver)?;
        }
        "sgd" => {
            let mut solver = SolverSGD::from_file(solver_cfg)?;
            solver.load_state(solver_state)?;
            test_net_helper(args, solver)?;
        }
        _ => {
            return Err(Box::new(CustomError::Other));
        }
    }

    Ok(())
}

fn test_net_helper(
    args: &ArgMatches,
    solver: impl Solver + Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ds_path = args.get_one::<String>("Data").unwrap();
    let ds = Box::new(ProtobufDataLoader::from_file(ds_path)?);
    let mut ds_test = Box::new(ProtobufDataLoader::from_file(ds_path)?);

    let test_batch = args.get_one::<usize>("TestBatch").unwrap();

    let mut net = Network::new(ds, solver);

    for i in 0..*test_batch {
        info!("Test {} , evaluating", i);
        let test_batch = ds_test.next();

        let mut label = -1;
        for (idx, it) in test_batch.expected.iter().enumerate() {
            if *it == 1.0 {
                label = idx as i32;
                break;
            }
        }

        info!("Below label is : {}", label);
        net.feedforward(&test_batch, true);
    }

    Ok(())
}
