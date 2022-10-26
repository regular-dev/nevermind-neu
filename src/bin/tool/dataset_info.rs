use regular_mind::dataloader::*;
use regular_mind::util::*;
use regular_mind::err::*;

use log::error;

use clap::ArgMatches;


pub fn dataset_info(args: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    if !args.contains_id("Data") {
        error!("Dataset wasn't provided (--dataset)");
        return Err(Box::new(CustomError::WrongArg));
    }

    let filepath = args.get_one::<String>("Data").unwrap();

    let loader = ProtobufDataLoader::from_file(filepath)?;
    println!("Dataset length : {}", loader.data.len());

    Ok(())
}

pub fn foo() {
    let val = 11531.0;
    let norm_val = minmax_normalize_val(val, -9200.0, 9200.0);
    println!("norm val : {}", norm_val);
}
