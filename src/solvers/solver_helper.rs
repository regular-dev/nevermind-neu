use std::collections::HashMap;

use uuid::Uuid;

use ndarray::Array2;
use std::str::FromStr;

use crate::dataloader::DataBatch;
use crate::layers_storage::LayersStorage;
use crate::solvers::pb::{PbFloatVec, PbWsBlob};
use crate::util::WsBlob;

pub fn feedforward(layers: &mut LayersStorage, train_data: &DataBatch, print_out: bool) {
    let mut out = None;

    for (idx, l) in layers.iter_mut().enumerate() {
        // handle input layer
        if idx == 0 {
            let result_out = l.forward_input(train_data.input.clone());

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
            prev_out = layers.at_mut(layers.len() - 2).learn_params();

            let result_out = layers
                .at_mut(layers.len() - 1)
                .backward_output(vec![prev_out.unwrap()], expected_data);

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

        let prev_out = layers.at_mut(layers.len() - 2 - idx).learn_params();
        let next_out = layers.at_mut(layers.len() - idx).learn_params();

        let result_out = layers
            .at_mut(layers.len() - 1 - idx)
            .backward(vec![prev_out.unwrap()], vec![next_out.unwrap()]);

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

pub fn convert_ws_blob_to_pb(ws_blob: &WsBlob) -> PbWsBlob {
    let mut pb_ws_blob = PbWsBlob::default();

    for i in ws_blob {
        let float_vec = PbFloatVec {
            vals: i.clone().into_raw_vec(),
            shape_size: i.shape()[0] as i32,
            shape_prev_size: i.shape()[1] as i32,
        };
        pb_ws_blob.ws.push(float_vec);
    }

    pb_ws_blob
}

pub fn convert_hash_ws_blob_to_pb(h: &HashMap<Uuid, WsBlob>) -> HashMap<String, PbWsBlob> {
    let mut out = HashMap::new();

    for (key, val) in h {
        out.insert(key.to_string(), convert_ws_blob_to_pb(val));
    }
    out
}

/// Deserializer function from Protobuf message to WsBlob
/// We need PbWsBlob as mutable to avoid unnecessary copying
pub fn convert_pb_to_ws_blob(pb_ws_blob: &mut PbWsBlob) -> WsBlob {
    let mut ws_blob = WsBlob::new();

    for it in pb_ws_blob.ws.iter_mut() {
        // TODO : handle unwrap
        let vals = std::mem::replace(&mut it.vals, Vec::new());
        let ws_mat =
            Array2::from_shape_vec((it.shape_size as usize, it.shape_prev_size as usize), vals)
                .unwrap();
        ws_blob.push(ws_mat);
    }

    return ws_blob;
}

pub fn convert_pb_to_hash_ws_blob(h: &mut HashMap<String, PbWsBlob>) -> HashMap<Uuid, WsBlob> {
    let mut out = HashMap::new();

    for (key, val) in h.iter_mut() {
        if let Ok(uid) = Uuid::from_str(key.as_str()) {
            out.insert(uid, convert_pb_to_ws_blob(val));
        }
    }

    out
}
