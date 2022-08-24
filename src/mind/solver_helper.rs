use super::dataset::DataBatch;
use super::layers_storage::LayersStorage;
use super::util::WsBlob;

pub fn feedforward(layers: &mut LayersStorage, train_data: &DataBatch, print_out: bool) {
    let input_data = &train_data.input;

    let mut out = None;

    for (idx, l) in layers.iter_mut().enumerate() {
        // handle input layer
        if idx == 0 {
            let result_out = l.forward(&vec![input_data]);

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

        let result_out = l.forward(&out.unwrap());

        match result_out {
            Err(_reason) => {
                return;
            }
            Ok(val) => {
                out = Some(val);
            }
        };
    }

    let out_val = &out.unwrap();

    if print_out {
        for i in out_val.iter() {
            println!("out val : {}", i);
        }
    }
}

pub fn backpropagate(layers: &mut LayersStorage, train_data: &DataBatch) {
    let expected_data = &train_data.expected;

    let mut out = None;

    for idx in 0..layers.len() {
        if idx == 0 {
            let prev_out = &layers.at(layers.len() - 2).learn_params().unwrap().output;

            let result_out = layers.at(idx).backward(Some(&vec![prev_out]), Some(&vec![expected_data]), None);

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

        let prev_out = &layers
            .at(layers.len() - 2 - idx)
            .learn_params()
            .unwrap()
            .output;

        let result_out = layers.at(idx).backward(
            Some(&vec![prev_out]),
            Some(&vec![out.unwrap().0]),
            Some(out.unwrap().1),
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
