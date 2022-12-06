use std::collections::HashMap;

use uuid::Uuid;

use ndarray::Array2;
use std::str::FromStr;

use crate::models::pb::{PbFloatVec, PbWsBlob};
use crate::util::WsBlob;


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
