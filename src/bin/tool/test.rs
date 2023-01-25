use log::{error, info};
use regular_mind::models::Model;
use regular_mind::models::Sequential;

use std::time::Instant;

use clap::ArgMatches;

// regular_mind
use regular_mind::dataloader::*;
use regular_mind::network::*;

pub fn test_net(
    args: &ArgMatches,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_cfg = args.get_one::<String>("ModelCfg").unwrap();
    let mut model = Sequential::from_file(&model_cfg)?;

    let model_state_file = args.get_one::<String>("ModelState").unwrap();
    model.load_state(&model_state_file)?;

    let ds_path = args.get_one::<String>("Data").unwrap();
    let test_data = Box::new(ProtobufDataLoader::from_file(ds_path)?);

    let test_batch = args.get_one::<usize>("TestBatch").unwrap();

    model.set_batch_size(1);

    let mut net = Network::new_for_eval(model);

    // let mut counter = 0;
    // TODO : check label impl
    // TODO : add features to display accuracy of testing
    for i in 0..*test_batch {
        info!("Test {} , evaluating", i);
        let test_batch = test_data.next_batch(1);

        let mut label = -1;
        for (idx, it) in test_batch.output.iter().enumerate() {
            if *it == 1.0 {
                label = idx as i32;
                break;
            }
        }
        
        if label != -1 {
            info!("Below label is : {}", label);
        }
        info!("Below output is : {}", test_batch.output);
        let out = net.eval(test_batch.input).unwrap();
        let out_b = out.borrow();
        info!("Neural output is : {}", out_b);
        info!("==========")
    }

    Ok(())
}