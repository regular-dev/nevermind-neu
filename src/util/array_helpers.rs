use ndarray::Array;

pub fn max<D>(arr: &Array<f32, D>) -> f32
where D: ndarray::Dimension
{
    let mut out = f32::MIN;
    
    for i in arr.iter() {
        if *i > out {
            out = *i;
        }
    }

    out
}

pub fn min<D>(arr: &Array<f32, D>) -> f32
where D: ndarray::Dimension
{
    let mut out = f32::MAX;

    for i in arr.iter() {
        if *i < out {
            out = *i;
        }
    }

    out
}