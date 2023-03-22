use log::{debug, error, info};
use signal_hook::consts::SIGKILL;

use std::{error::Error, time::Instant};

use signal_hook::{consts::SIGINT, iterator::Signals};

use clap::ArgMatches;

// nevermind_neu
use nevermind_neu::dataloader::*;
use nevermind_neu::err::*;
use nevermind_neu::models::*;
use nevermind_neu::orchestra::*;
use nevermind_neu::optimizers::*;


pub fn train_net_ocl(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let mut net = create_net_ocl_from_cmd_args(args)?;

    //set_train_dataset_to_net(&mut net, args)?;

    let mut opt_err = None;
    let mut opt_max_iter = None;

    if let Some(err) = args.get_one::<f32>("Err") {
        info!("Satisfying error : {}", err);
        opt_err = Some(err);
    }

    if let Some(max_iter) = args.get_one::<usize>("MaxIter") {
        info!("Iteration limit : {}", max_iter);
        opt_max_iter = Some(max_iter);
    }

    let now_time = Instant::now();

    if opt_err.is_some() && opt_max_iter.is_some() {
        let err = opt_err.unwrap();
        let max_iter = opt_max_iter.unwrap();

        info!(
            "Start train till the err {} or max iteration {}",
            *err, *max_iter
        );
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

pub fn create_net_ocl_from_cmd_args(args: &ArgMatches) -> Result<Orchestra<SequentialOcl>, Box<dyn Error>> {
    let model_cfg = args.get_one::<String>("ModelCfg").unwrap();
    let mut model = SequentialOcl::from_file(&model_cfg)?;

    if let Some(model_state) = args.get_one::<String>("ModelState") {
        model.load_state(&model_state)?;
    }

    if let Some(optimizer_cfg) = args.get_one::<String>("OptCfg") {
        info!("Setting up optimizer : {}", optimizer_cfg);
        let opt = optimizer_ocl_from_file(optimizer_cfg, model.queue())?;
        model.set_optim(opt);
    }

    let train_ds = args.get_one::<String>("TrainData").unwrap();
    let train_ds = Box::new(ProtobufDataLoader::from_file(train_ds)?);

    info!("Train batch size : {}", model.batch_size());

    let mut net = Orchestra::new(model);

    net.set_train_dataset(train_ds);

    let mut signals = Signals::new(&[SIGINT])?;

    net.add_callback(Box::new(move |_, _, _| {
        for sig in signals.pending() {
            debug!("Received signal {:?}", sig);

            if sig == SIGINT {
                return CallbackReturnAction::StopAndSave;
            } else if sig == SIGKILL {
                return CallbackReturnAction::Stop;
            }
        }

        CallbackReturnAction::None
    }));

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
        net = net.test_batch_size(*test_batch);
    }

    if let Some(lr_decay) = args.get_one::<f32>("LrDecay") {
        info!("Setting learning rate decay to : {}", lr_decay);
        net.set_learn_rate_decay(*lr_decay);
    }

    if let Some(lr_step) = args.get_one::<usize>("LrStep") {
        info!("Setting learning rate decay step to : {}", lr_step);
        net.set_learn_rate_decay_step(*lr_step);
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

    Ok(net)
}
