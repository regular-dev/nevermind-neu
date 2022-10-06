use ndarray::Zip;

use crate::util::DataVec;

/// With minmax normalising values will be between 0..1
pub fn minmax_normalize(data: &mut DataVec) {
    let min = data.fold(f32::MAX, 
    |val_min, val| {
        if *val < val_min {
            return *val;
        } else {
            return val_min;
        }
    });

    let max = data.fold(f32::MIN,
    |val_max, val| {
        if *val > val_max {
            return *val;
        } else {
            return val_max;
        }
    });

    minmax_normalize_params(data, min, max);
}

pub fn minmax_normalize_val(val: f32, min: f32, max: f32) -> f32 {
    (val - min) / (max - min)
}

pub fn minmax_normalize_params(data: &mut DataVec, min: f32, max: f32) {
    Zip::from(data).for_each(
        |el| {
            *el = minmax_normalize_val(*el, min, max);
        }
    );
}