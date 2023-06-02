use std::{
    cell::RefCell,
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    sync::Arc,
};

use log::debug;

use ndarray_rand::rand::rngs::SmallRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use ndarray_rand::RandomExt;

use super::util::{Array1D, Array2D, Float, WsBlob, WsMat};

#[derive(PartialEq)]
#[repr(i32)]
pub enum TypeBuffer {
    Weights = 0,
    WeightsGrad = 1,
    Output = 2,
    Bias = 3,
    NeuGrad = 4,
    BiasGrad = 5, // averaged neuron gradient
}

#[derive(Clone)]
pub enum VariantParamArc {
    Array1(Arc<RefCell<Array1D>>),
    Array2(Arc<RefCell<Array2D>>),
}

impl VariantParamArc {
    pub fn get_arr_1d(&self) -> Arc<RefCell<Array1D>> {
        if let VariantParamArc::Array1(a) = self {
            return a.clone();
        } else { 
            panic!("VariantParamArc is not Array1");
        }
    }

    pub fn get_arr_2d(&self) -> Arc<RefCell<Array2D>> {
        if let VariantParamArc::Array2(a) = self {
            return a.clone();
        } else { 
            panic!("VariantParamArc is not Array2");
        }
    }}

#[derive(Clone)]
pub enum VariantParam {
    Array1(Array1D),
    Array2(Array2D),
}

impl VariantParam {
    pub fn copy_zeroed_shape_from(arg: &VariantParamArc) -> Self {
        match arg {
            VariantParamArc::Array1(arr1) => {
                let arr1_bor = arr1.borrow();
                return VariantParam::Array1(Array1D::zeros(arr1_bor.len()));
            }
            VariantParamArc::Array2(arr2) => {
                let arr2_bor = arr2.borrow();
                return VariantParam::Array2(Array2D::zeros((
                    arr2_bor.shape()[0],
                    arr2_bor.shape()[1],
                )));
            }
        }
    }
}

impl From<i32> for TypeBuffer {
    fn from(value: i32) -> Self {
        if value == 0 {
            return TypeBuffer::Weights;
        } else if value == 1 {
            return TypeBuffer::WeightsGrad;
        } else if value == 2 {
            return TypeBuffer::Output;
        } else if value == 3 {
            return TypeBuffer::Bias;
        } else if value == 4 {
            return TypeBuffer::NeuGrad;
        } else if value == 5 {
            return TypeBuffer::BiasGrad;
        } else {
            panic!("Invalid integer to convert");
        }
    }
}

impl VariantParamArc {
    fn copy(&self) -> VariantParamArc {
        match self {
            VariantParamArc::Array1(arr) => {
                let arr_b = arr.borrow().clone();
                return VariantParamArc::Array1(Arc::new(RefCell::new(arr_b)));
            }
            VariantParamArc::Array2(arr) => {
                let arr_b = arr.borrow().clone();
                return VariantParamArc::Array2(Arc::new(RefCell::new(arr_b)));
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct CpuParams {
    pub params: HashMap<i32, VariantParamArc>, // pub may be removed further
    pub id: u64,
}

pub type LearnParamsPtr = Arc<RefCell<CpuParams>>;
pub type ParamsBlob = Vec<CpuParams>;

impl CpuParams {
    pub fn new(size: usize, prev_size: usize) -> Self {
        let ws = VariantParamArc::Array2(Arc::new(RefCell::new(WsMat::random(
            (size, prev_size),
            Uniform::new(-0.1, 0.1),
        ))));
        let ws_grad =
            VariantParamArc::Array2(Arc::new(RefCell::new(WsMat::zeros((size, prev_size)))));
        let output = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::zeros((1, size)))));
        let neu_grad = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::zeros((1, size)))));

        let mut m = HashMap::new();
        m.insert(TypeBuffer::Output as i32, output);
        m.insert(TypeBuffer::Weights as i32, ws);
        m.insert(TypeBuffer::WeightsGrad as i32, ws_grad);
        m.insert(TypeBuffer::NeuGrad as i32, neu_grad);

        Self {
            params: m,
            id: CpuParams::generate_u64_id(),
        }
    }

    pub fn empty() -> Self {
        Self {
            params: HashMap::new(),
            id: CpuParams::generate_u64_id(),
        }
    }

    pub fn new_only_output(size: usize) -> Self {
        let output = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::zeros((1, size)))));

        let mut m = HashMap::new();
        m.insert(TypeBuffer::Output as i32, output);

        Self {
            params: m,
            id: CpuParams::generate_u64_id(),
        }
    }

    pub fn new_with_bias(size: usize, prev_size: usize) -> Self {
        let ws = VariantParamArc::Array2(Arc::new(RefCell::new(WsMat::random(
            (size, prev_size),
            Uniform::new(-0.1, 0.1),
        ))));
        let ws_grad =
            VariantParamArc::Array2(Arc::new(RefCell::new(WsMat::zeros((size, prev_size)))));
        let bias = VariantParamArc::Array1(Arc::new(RefCell::new(Array1D::random(
            size,
            Uniform::new(-0.1, 0.1),
        ))));
        let neu_grad = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::random(
            (1, size),
            Uniform::new(-0.1, 0.1),
        ))));
        let output = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::zeros((1, size)))));
        let bias_grad = VariantParamArc::Array1(Arc::new(RefCell::new(Array1D::zeros(size))));

        let mut m = HashMap::new();
        m.insert(TypeBuffer::Output as i32, output);
        m.insert(TypeBuffer::Weights as i32, ws);
        m.insert(TypeBuffer::WeightsGrad as i32, ws_grad);
        m.insert(TypeBuffer::Bias as i32, bias);
        m.insert(TypeBuffer::NeuGrad as i32, neu_grad);
        m.insert(TypeBuffer::BiasGrad as i32, bias_grad);

        Self {
            params: m,
            id: CpuParams::generate_u64_id(),
        }
    }

    pub fn new_with_const_bias(size: usize, prev_size: usize, bias_val: f32) -> Self {
        let ws = VariantParamArc::Array2(Arc::new(RefCell::new(WsMat::random(
            (size, prev_size),
            Uniform::new(-0.1, 0.1),
        ))));
        let ws_grad =
            VariantParamArc::Array2(Arc::new(RefCell::new(WsMat::zeros((size, prev_size)))));
        let bias =
            VariantParamArc::Array1(Arc::new(RefCell::new(Array1D::from_elem(size, bias_val))));
        let neu_grad = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::zeros((1, size)))));
        let output = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::zeros((1, size)))));
        let bias_grad = VariantParamArc::Array1(Arc::new(RefCell::new(Array1D::zeros(size))));

        let mut m = HashMap::new();
        m.insert(TypeBuffer::Output as i32, output);
        m.insert(TypeBuffer::Weights as i32, ws);
        m.insert(TypeBuffer::WeightsGrad as i32, ws_grad);
        m.insert(TypeBuffer::Bias as i32, bias);
        m.insert(TypeBuffer::NeuGrad as i32, neu_grad);
        m.insert(TypeBuffer::BiasGrad as i32, bias_grad);

        Self {
            params: m,
            id: CpuParams::generate_u64_id(),
        }
    }

    pub fn get_1d_buf(&self, id: i32) -> Arc<RefCell<Array1D>> {
        let res_prm = self.params.get(&id).unwrap();

        if let VariantParamArc::Array1(arr) = res_prm {
            return arr.clone();
        } else {
            panic!("No Array1D with id {} was found", id);
        }
    }

    pub fn get_1d_buf_t(&self, id: TypeBuffer) -> Arc<RefCell<Array1D>> {
        return self.get_1d_buf(id as i32);
    }

    pub fn get_2d_buf(&self, id: i32) -> Arc<RefCell<Array2D>> {
        let res_prm = self.params.get(&id).unwrap();

        if let VariantParamArc::Array2(arr) = res_prm {
            return arr.clone();
        } else {
            panic!("No Array2D with id {} was found", id);
        }
    }

    pub fn get_2d_buf_t(&self, id: TypeBuffer) -> Arc<RefCell<Array2D>> {
        return self.get_2d_buf(id as i32);
    }

    pub fn get_param(&self, id: i32) -> VariantParamArc {
        return self.params.get(&id).unwrap().clone();
    }

    pub fn get_param_t(&self, id: TypeBuffer) -> VariantParamArc {
        return self.params.get(&(id as i32)).unwrap().clone();
    }

    pub fn insert_buf(&mut self, id: i32, p: VariantParamArc) {
        self.params.insert(id, p);
    }

    pub fn remove_buf(&mut self, id: i32) {
        self.params.remove(&id);
    }

    pub fn contains_buf(&self, id: i32) -> bool {
        self.params.contains_key(&id)
    }

    pub fn contains_buf_t(&self, id: TypeBuffer) -> bool {
        self.contains_buf(id as i32)
    }

    pub fn fit_to_batch_size(&mut self, new_batch_size: usize) {
        let out_m = self.get_2d_buf_t(TypeBuffer::Output);
        let mut out_m = out_m.borrow_mut();
        let size = out_m.ncols();

        if out_m.nrows() != new_batch_size {
            *out_m = Array2D::zeros((new_batch_size, size));

            if self.contains_buf_t(TypeBuffer::NeuGrad) {
                let err_m = self.get_2d_buf_t(TypeBuffer::NeuGrad);
                let mut err_m = err_m.borrow_mut();
                *err_m = Array2D::zeros((new_batch_size, size));
            }
        }
    }

    pub fn prepare_for_tests(&mut self, batch_size: usize) {
        let out_size = self.get_2d_buf_t(TypeBuffer::Output).borrow().ncols();

        self.params.remove(&(TypeBuffer::NeuGrad as i32));
        self.params.remove(&(TypeBuffer::Output as i32));

        let output = VariantParamArc::Array2(Arc::new(RefCell::new(Array2D::zeros((
            batch_size, out_size,
        )))));

        self.params.insert(TypeBuffer::Output as i32, output);
    }

    /// Copies learn_params memory of (weights, gradients, output...)
    /// This function DO copy memory
    /// To clone only Arc<...> use .clone() function
    pub fn copy(&self) -> Self {
        let mut lp = CpuParams::default();

        {
            for (k, v) in self.params.iter() {
                let v_copied = v.copy();
                lp.params.insert(*k, v_copied);
            }

            lp.id = self.id.clone();
        }

        lp
    }

    /// Could be used also for OclParams and others
    pub fn generate_u64_id() -> u64 {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }
}
