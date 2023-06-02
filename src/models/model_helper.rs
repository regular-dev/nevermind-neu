use std::collections::HashMap;

use ndarray::Array2;
use std::{str::FromStr, sync::Arc, cell::RefCell};

use crate::cpu_params::{VariantParamArc, TypeBuffer};
use crate::models::pb::{PbBuf, PbBufBlob};
use crate::util::*;

pub fn convert_buf_2d_to_pb(buf: &Array2D, id: i32) -> PbBuf {
    let mut pb_ws_blob = PbBuf::default();

    pb_ws_blob.vals = buf.clone().into_raw_vec();
    pb_ws_blob.buf_id = id;

    for s_i in buf.shape() {
        pb_ws_blob.shape.push(*s_i as i32);
    }

    pb_ws_blob
}

pub fn convert_buf_1d_to_pb(buf: &Array1D, id: i32) -> PbBuf {
    let mut pb_ws_blob = PbBuf::default();

    pb_ws_blob.vals = buf.clone().into_raw_vec();
    pb_ws_blob.buf_id = id;

    for s_i in buf.shape() {
        pb_ws_blob.shape.push(*s_i as i32);
    }

    pb_ws_blob
}

pub fn convert_pb_to_param_buf(pb_buf: &PbBuf) -> VariantParamArc {
    if pb_buf.shape.len() == 1 {
        let arr1 = Array1D::from_shape_vec(pb_buf.shape[0] as usize, pb_buf.vals.clone()).expect("Deserialize Array1D from protobuf");
        return VariantParamArc::Array1(Arc::new(RefCell::new(arr1)));
    } else if pb_buf.shape.len() == 2 {
        let arr2 = Array2D::from_shape_vec((pb_buf.shape[0] as usize, pb_buf.shape[1] as usize), pb_buf.vals.clone()).expect("Deserialize Array2D from protobuf");
        return VariantParamArc::Array2(Arc::new(RefCell::new(arr2)));
    } else {
        panic!("Invalid shape, deserializing protobuf");
    }
}

// pub fn convert_hash_ws_blob_to_pb(h: &HashMap<Uuid, WsBlob>) -> HashMap<String, PbWsBlob> {
//     let mut out = HashMap::new();

//     for (key, val) in h {
//         out.insert(key.to_string(), convert_buf_2d_to_pb(val));
//     }
//     out
// }

// Deserializer function from Protobuf message to WsBlob
// We need PbWsBlob as mutable to avoid unnecessary copying
// pub fn convert_pb_to_ws_blob(pb_ws_blob: &mut PbWsBlob) -> WsBlob {
//     let mut ws_blob = WsBlob::new();

//     for it in pb_ws_blob.ws.iter_mut() {
//         // TODO : handle unwrap
//         let vals = std::mem::replace(&mut it.vals, Vec::new());
//         let vals_len = vals.len();

//         let ws_mat =
//             Array2::from_shape_vec((it.shape_size as usize, it.shape_prev_size as usize), vals)
//                 .expect(
//                     format!(
//                         "Invalid weights shape. Given : {}, Self : {} / {}",
//                         vals_len, it.shape_size, it.shape_prev_size
//                     )
//                     .as_str(),
//                 );
//         ws_blob.push(ws_mat);
//     }

//     return ws_blob;
// }

// pub fn convert_pb_to_hash_ws_blob(h: &mut HashMap<String, PbWsBlob>) -> HashMap<Uuid, WsBlob> {
//     let mut out = HashMap::new();

//     for (key, val) in h.iter_mut() {
//         if let Ok(uid) = Uuid::from_str(key.as_str()) {
//             out.insert(uid, convert_pb_to_ws_blob(val));
//         }
//     }

//     out
// }
