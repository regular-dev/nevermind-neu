use super::dataset::DataBatch;
use super::layers_storage::LayersStorage;
use super::util::WsBlob;

pub fn feedforward(layers: &mut LayersStorage, train_data: &DataBatch, print_out: bool) {
    let input_data = &train_data.input;

    let mut out = None;

    for (idx, l) in layers.iter_mut().enumerate() {
        // handle input layer
        if idx == 0 {
            let result_out = l.forward_input(&input_data);

            match result_out {
                Err(_reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            };
            continue;
        }

        let result_out = l.forward(out.unwrap());

        match result_out {
            Err(_reason) => {
                return;
            }
            Ok(val) => {
                out = Some(val);
            }
        };
    }

    let out_v = out.as_ref().unwrap()[0].output.borrow();
    // let out_val = out_v.output.borrow();

    if print_out {
        for i in out_v.iter() {
            println!("out val : {}", i);
        }
    }
}

pub fn backpropagate(layers: &mut LayersStorage, train_data: &DataBatch) {
    let expected_data = &train_data.expected;

    let mut prev_out = None;
    let mut out = None;

    for idx in 0..layers.len() {
        if idx == 0 {
            prev_out = layers.at(layers.len() - 2).learn_params();

            let result_out = layers.at(layers.len() - 1).backward_output(vec![prev_out.unwrap()],
                                                            expected_data);

            match result_out {
                Err(reason) => {
                    return;
                }
                Ok(val) => {
                    out = Some(val);
                }
            }
            continue;
        }

        if idx == layers.len() - 1 {
            continue;
        }

        let prev_out = layers.at(layers.len() - 2 - idx).learn_params();
        let next_out = layers.at(layers.len() - idx).learn_params();

        let result_out = layers.at(idx).backward(
            vec![prev_out.unwrap()],
            vec![next_out.unwrap()] 
        );

        match result_out {
            Err(_reason) => {
                return;
            }
            Ok(val) => {
                out = Some(val);
            }
        }
    }
}
