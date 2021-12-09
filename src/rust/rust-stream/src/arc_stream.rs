use std::iter::Sum;
use std::sync::{Arc, Mutex};

use self::core_affinity::CoreId;
use crate::stream::{AllocatorType, ArrayType, RustStream, StreamData};

struct ArcHeapData<T: ArrayType, A: AllocatorType> {
  a_chunks: Vec<Arc<Mutex<Vec<T, A>>>>,
  b_chunks: Vec<Arc<Mutex<Vec<T, A>>>>,
  c_chunks: Vec<Arc<Mutex<Vec<T, A>>>>,
}

pub struct ArcDevice<T: ArrayType, A: AllocatorType> {
  pub(crate) ncore: usize,
  pub(crate) pin: bool,
  pub(crate) core_ids: Vec<CoreId>,
  data: ArcHeapData<T, A>,
}

impl<T: ArrayType, A: AllocatorType> ArcDevice<T, A> {
  pub fn new(ncore: usize, pin: bool, alloc: A) -> Self {
    let mut core_ids = match core_affinity::get_core_ids() {
      Some(xs) => xs,
      None => {
        colour::e_red_ln!("Cannot enumerate cores, pinning will not work if enabled");
        (0..ncore).map(|i| CoreId { id: i }).collect()
      }
    };
    core_ids.resize(ncore, core_ids[0]);

    let lift =
      || (0..ncore).map(|_| return Arc::new(Mutex::new(Vec::new_in(alloc)))).collect::<Vec<_>>();
    let data = ArcHeapData { a_chunks: lift(), b_chunks: lift(), c_chunks: lift() };

    ArcDevice { ncore, pin, core_ids, data }
  }

  pub fn ref_a(&self, t: usize) -> Arc<Mutex<Vec<T, A>>> { self.data.a_chunks[t].clone() }

  pub fn ref_b(&self, t: usize) -> Arc<Mutex<Vec<T, A>>> { self.data.b_chunks[t].clone() }

  pub fn ref_c(&self, t: usize) -> Arc<Mutex<Vec<T, A>>> { self.data.c_chunks[t].clone() }

  // divide the length by the number of cores, the last core gets less work if it does not divide
  fn chunk_size(&self, len: usize, t: usize) -> usize {
    assert!(t < self.ncore);
    let chunk = (len as f64 / self.ncore as f64).ceil() as usize;
    if t == self.ncore - 1 {
      len - (t * chunk)
    } else {
      chunk
    }
  }
}

extern crate core_affinity;

// Arc+Mutex threaded version, it should be semantically equal to the single threaded version
impl<T: 'static + ArrayType + Sync + Send + Sum, A: AllocatorType + Sync + Send + 'static>
  RustStream<T> for StreamData<T, ArcDevice<T, A>, A>
{
  fn init_arrays(&mut self) {
    let init = self.init;
    let pin = self.device.pin;
    (0..self.device.ncore)
      .map(&|t| {
        let ref_a = self.device.ref_a(t);
        let ref_b = self.device.ref_b(t);
        let ref_c = self.device.ref_c(t);
        let core = self.device.core_ids[t];
        let n = self.device.chunk_size(self.size, t);
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          ref_a.lock().unwrap().resize(n, init.0);
          ref_b.lock().unwrap().resize(n, init.1);
          ref_c.lock().unwrap().resize(n, init.2);
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .for_each(|t| t.join().unwrap());
  }
  fn read_arrays(&mut self) {
    let range = self.size;
    let unlift = |drain: &mut Vec<T, A>, source: &Vec<Arc<Mutex<Vec<T, A>>>>| {
      let xs =
        source.into_iter().flat_map(|x| x.lock().unwrap().clone().into_iter()).collect::<Vec<_>>();
      for i in 0..range {
        drain[i] = xs[i];
      }
    };
    unlift(&mut self.a, &self.device.data.a_chunks);
    unlift(&mut self.b, &self.device.data.b_chunks);
    unlift(&mut self.c, &self.device.data.c_chunks);
  }

  fn copy(&mut self) {
    let pin = self.device.pin;
    (0..self.device.ncore)
      .map(move |t| {
        let ref_a = self.device.ref_a(t);
        let ref_c = self.device.ref_c(t);
        let core = self.device.core_ids[t];
        let n = self.device.chunk_size(self.size, t);
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = ref_a.lock().unwrap();
          let mut c = ref_c.lock().unwrap();
          for i in 0..n {
            c[i] = a[i];
          }
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .for_each(|t| t.join().unwrap());
  }

  fn mul(&mut self) {
    let scalar = self.scalar;
    let pin = self.device.pin;
    (0..self.device.ncore)
      .map(move |t| {
        let ref_b = self.device.ref_b(t);
        let ref_c = self.device.ref_c(t);
        let core = self.device.core_ids[t];
        let n = self.device.chunk_size(self.size, t);
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let mut b = ref_b.lock().unwrap();
          let c = ref_c.lock().unwrap();
          for i in 0..n {
            b[i] = scalar * c[i];
          }
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .for_each(|t| t.join().unwrap());
  }

  fn add(&mut self) {
    let pin = self.device.pin;
    (0..self.device.ncore)
      .map(&|t| {
        let ref_a = self.device.ref_a(t);
        let ref_b = self.device.ref_b(t);
        let ref_c = self.device.ref_c(t);
        let core = self.device.core_ids[t];
        let n = self.device.chunk_size(self.size, t);
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = ref_a.lock().unwrap();
          let b = ref_b.lock().unwrap();
          let mut c = ref_c.lock().unwrap();
          for i in 0..n {
            c[i] = a[i] + b[i];
          }
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .for_each(|t| t.join().unwrap());
  }

  fn triad(&mut self) {
    let scalar = self.scalar;
    let pin = self.device.pin;
    (0..self.device.ncore)
      .map(&|t| {
        let ref_a = self.device.ref_a(t);
        let ref_b = self.device.ref_b(t);
        let ref_c = self.device.ref_c(t);
        let core = self.device.core_ids[t];
        let n = self.device.chunk_size(self.size, t);
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let mut a = ref_a.lock().unwrap();
          let b = ref_b.lock().unwrap();
          let c = ref_c.lock().unwrap();
          for i in 0..n {
            a[i] = b[i] + scalar * c[i]
          }
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .for_each(|t| t.join().unwrap());
  }

  fn nstream(&mut self) {
    let scalar = self.scalar;
    let pin = self.device.pin;
    (0..self.device.ncore)
      .map(&|t| {
        let ref_a = self.device.ref_a(t);
        let ref_b = self.device.ref_b(t);
        let ref_c = self.device.ref_c(t);
        let core = self.device.core_ids[t];
        let n = self.device.chunk_size(self.size, t);
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let mut a = ref_a.lock().unwrap();
          let b = ref_b.lock().unwrap();
          let c = ref_c.lock().unwrap();
          for i in 0..n {
            a[i] += b[i] + scalar * c[i]
          }
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .for_each(|t| t.join().unwrap());
  }

  fn dot(&mut self) -> T {
    let pin = self.device.pin;
    (0..self.device.ncore)
      .map(&|t| {
        let ref_a = self.device.ref_a(t);
        let ref_b = self.device.ref_b(t);
        let core = self.device.core_ids[t];
        let n = self.device.chunk_size(self.size, t);
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = ref_a.lock().unwrap();
          let b = ref_b.lock().unwrap();
          let mut p = T::default();
          for i in 0..n {
            p += a[i] * b[i];
          }
          p
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .map(|t| t.join().unwrap())
      .sum()
  }
}
