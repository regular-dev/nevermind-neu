use std::{cell::RefCell, collections::HashMap, error::Error, rc::Rc};

use tui::widgets::canvas::Shape;

use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ocl::{Buffer, Context, Device, MemFlags, ProQue, Queue};

use log::warn;

use crate::layers::*;
use crate::cpu_params::*;
use crate::models::pb::*;
use crate::util::*;

pub type LayerOclResult = Result<Vec<OclParams>, LayerError>;
pub type OclWsBlob = Vec<Buffer<Float>>;
pub type OclBufRc = Rc<RefCell<Buffer<Float>>>;
pub type ShapeVec = Vec<i32>; // usize ?
pub type OclShapeBuf = (OclBufRc, ShapeVec);

#[derive(Clone, Default)]
pub struct OclParams {
    pub params: HashMap<i32, OclShapeBuf>,
    pub id: u64,
}

pub type OclParamsBlob = Vec<OclParams>;

pub enum FetchParams {
    Output,
    Ws,
    NeuGrad,
    WsGrad,
    OutputAndNeuGrad,
    All,
}

impl OclParams {
    pub fn only_output(buf: Buffer<f32>, queue: Queue) -> Self {
        let shape = buf.len();
        let output = Rc::new(RefCell::new(buf));
        let mut ocl_params = OclParams::default();

        ocl_params.insert_buf(TypeBuffer::Output as i32, output, vec![shape as i32]);
        ocl_params.id = CpuParams::generate_u64_id();

        return ocl_params;
    }

    pub fn serialize_to_pb(&self, ser_ids: &[i32]) -> PbBufBlob {
        let mut pb_ws = PbBufBlob::default();

        for ser_id in ser_ids.iter() {
            let shape_buf = self.get_buf(*ser_id);
            let buf_bor = shape_buf.0.borrow();
            let mut vec_buf = vec![0.0; buf_bor.len()];

            buf_bor
                .read(&mut vec_buf)
                .enq()
                .expect(format!("Failed to read from ocl {} buffer", *ser_id).as_str());
            pb_ws.bufs.push(PbBuf {
                vals: vec_buf,
                shape: shape_buf.1,
                buf_id: *ser_id,
            });
        }

        pb_ws
    }

    pub fn set_vals_from_pb(&mut self, buf_blob: &PbBufBlob, q: Queue) {
        for b_i in buf_blob.bufs.iter() {
            let mut final_len = 1;

            if b_i.shape.is_empty() {
                warn!("[ocl] Protobuf shape is empty");
            }

            for i in b_i.shape.iter() {
                if *i == 0 {
                    warn!("[ocl] Shape size equals to {}", i);
                }

                final_len = final_len * i;
            }

            let buf = Buffer::builder()
                .queue(q.clone())
                .flags(MemFlags::new().read_write())
                .len(final_len)
                .copy_host_slice(b_i.vals.as_slice())
                .build()
                .expect("Failed to create buffer from protobuf");

            self.insert_buf(b_i.buf_id, Rc::new(RefCell::new(buf)), b_i.shape.clone());
        }
    }

    pub fn set_ws_from_vec(&mut self, vals: &mut Vec<Float>, shape: ShapeVec, q: Queue) {
        let mut final_len = 1;

        if shape.is_empty() {
            warn!("[ocl] Shape is empty");
            final_len = 0;
        }

        for i in shape.iter() {
            if *i == 0 {
                warn!("[ocl] Shape equals {}", i);
            }

            final_len = i * final_len;
        }

        let ws = Buffer::builder()
            .queue(q)
            .flags(MemFlags::new().read_write())
            .len(final_len)
            .copy_host_slice(vals.as_slice())
            .build()
            .expect("Failed to create weights buffer from vec");

        self.insert_buf(TypeBuffer::Weights as i32, Rc::new(RefCell::new(ws)), shape);
    }

    pub fn set_bias_from_vec(&mut self, v: &mut Vec<Float>, q: Queue) {
        let bias = Buffer::builder()
            .queue(q)
            .flags(MemFlags::new().read_write())
            .len(v.len())
            .copy_host_slice(v.as_slice())
            .build()
            .expect("Failed to create bias buffer from vec");

        self.insert_buf(
            TypeBuffer::Bias as i32,
            Rc::new(RefCell::new(bias)),
            vec![v.len() as i32],
        );
    }

    pub fn fit_to_batch_size_ocl(
        &mut self,
        self_size: usize,
        batch_size: usize,
        queue: Queue,
    ) -> Result<(), Box<dyn Error>> {
        let output = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_write())
            .len(self_size * batch_size)
            .build()?;
        let neu_grad = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_write())
            .len(self_size * batch_size)
            .build()?;

        self.insert_buf(
            TypeBuffer::Output as i32,
            Rc::new(RefCell::new(output)),
            vec![self_size as i32, batch_size as i32],
        );
        self.insert_buf(
            TypeBuffer::NeuGrad as i32,
            Rc::new(RefCell::new(neu_grad)),
            vec![self_size as i32, batch_size as i32],
        );

        Ok(())
    }

    pub fn insert_buf(&mut self, id: i32, b: OclBufRc, s: ShapeVec) {
        self.params.insert(id, (b, s));
    }

    pub fn remove_buf(&mut self, id: i32) {
        self.params.remove(&id);
    }

    pub fn get_buf(&self, id: i32) -> OclShapeBuf {
        self.params[&id].clone()
    }

    pub fn get_buf_t(&self, id: TypeBuffer) -> OclShapeBuf {
        return self.get_buf(id as i32);
    }

    pub fn empty() -> Self {
        return Self::default();
    }

    pub fn create_empty_buf(
        l: usize,
        flags: MemFlags,
        q: Queue,
    ) -> Result<Buffer<Float>, Box<dyn Error>> {
        let output = Buffer::builder().queue(q).flags(flags).len(l).build()?;

        return Ok(output);
    }
}

pub fn init_ocl_params(
    queue: Queue,
    self_size: usize,
    prev_shape: &[usize],
    add_bias: bool,
) -> Result<OclParams, Box<dyn Error>> {
    let output = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size)
        .build()?;
    let neu_grad = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size)
        .build()?;
    let ws_grad = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size * prev_shape[0])
        .build()?;

    let ws_cpu_vals = WsMat::random((self_size, prev_shape[0]), Uniform::new(-0.1, 0.1));

    let ws = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(self_size * prev_shape[0])
        .copy_host_slice(ws_cpu_vals.as_slice().unwrap())
        .build()?;

    let mut ocl_params = OclParams {
        params: HashMap::new(),
        id: CpuParams::generate_u64_id(),
    };

    // Insert buffers
    ocl_params.insert_buf(
        TypeBuffer::Weights as i32,
        Rc::new(RefCell::new(ws)),
        vec![self_size as i32, prev_shape[0] as i32],
    );
    ocl_params.insert_buf(
        TypeBuffer::WeightsGrad as i32,
        Rc::new(RefCell::new(ws_grad)),
        vec![self_size as i32, prev_shape[0] as i32],
    );
    ocl_params.insert_buf(
        TypeBuffer::Output as i32,
        Rc::new(RefCell::new(output)),
        vec![self_size as i32],
    );
    ocl_params.insert_buf(
        TypeBuffer::NeuGrad as i32,
        Rc::new(RefCell::new(neu_grad)),
        vec![self_size as i32, prev_shape[0] as i32],
    );

    // Add bias buffers if necessary
    if add_bias {
        let bias_cpu_vals = ndarray::Array1::<Float>::random(self_size, Uniform::new(-0.1, 0.1));

        let bias = Buffer::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().read_write())
            .len(self_size)
            .copy_host_slice(bias_cpu_vals.as_slice().unwrap())
            .build()?;

        ocl_params.insert_buf(
            TypeBuffer::Bias as i32,
            Rc::new(RefCell::new(bias)),
            vec![self_size as i32],
        );
    }

    Ok(ocl_params)
}
