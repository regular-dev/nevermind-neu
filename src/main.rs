mod mind;

use crate::mind::network::Network;
use crate::mind::dataset::DataBatch;
use crate::mind::dataset::SimpleDataLoader;

fn main() {
    // prepare data set
    let mut dataset: Vec<DataBatch> = Vec::new();
    dataset.push( DataBatch::new(vec![0.0, 0.0], vec![0.0]) );
    dataset.push( DataBatch::new(vec![0.0, 1.0], vec![1.0]) );
    dataset.push( DataBatch::new(vec![1.0, 0.0], vec![1.0]) );
    dataset.push( DataBatch::new(vec![1.0, 1.0], vec![0.0]) );

    let dataloader = Box::new(SimpleDataLoader::new(dataset));

    // create a network
    let mut net = Network::new(dataloader);
    let net_cfg = vec![2, 5, 1];
    net.setup_network(&net_cfg);
    net.perform_step();
}
