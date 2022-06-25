mod mind;

use crate::mind::network::Network;
use crate::mind::dataset::DataBatch;
use crate::mind::dataset::SimpleDataLoader;

fn main() {
    // prepare data set
    let mut dataset_train: Vec<DataBatch> = Vec::new();
    dataset_train.push( DataBatch::new(vec![0.0, 0.0, 1.0], vec![0.0]) );
    dataset_train.push( DataBatch::new(vec![0.0, 1.0, 1.0], vec![1.0]) );
    dataset_train.push( DataBatch::new(vec![1.0, 0.0, 1.0], vec![1.0]) );
    dataset_train.push( DataBatch::new(vec![1.0, 1.0, 1.0], vec![0.0]) );

    let dataloader = Box::new(SimpleDataLoader::new(dataset_train));

    // create a network
    let mut net = Network::new(dataloader);
    let net_cfg = vec![3, 12, 5, 1];
    net.setup_network(&net_cfg);

    net.train_for_n_times(10000);

    // test dataset
    let mut dataset_test: Vec<DataBatch> = Vec::new();
    dataset_test.push( DataBatch::new(vec![0.0, 0.0, 1.0], vec![0.0]) );
    dataset_test.push( DataBatch::new(vec![0.0, 1.0, 1.0], vec![1.0]) );
    dataset_test.push( DataBatch::new(vec![1.0, 0.0, 1.0], vec![1.0]) );
    dataset_test.push( DataBatch::new(vec![1.0, 1.0, 1.0], vec![0.0]) );

    println!("Now testing net !!!");

    net.feedforward(&dataset_test[0], true);
    net.feedforward(&dataset_test[1], true);
    net.feedforward(&dataset_test[2], true);
    net.feedforward(&dataset_test[3], true);
}
