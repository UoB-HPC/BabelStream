use std::iter::Sum;
use std::slice::{Chunks, ChunksMut};

use crossbeam::thread;

use self::core_affinity::CoreId;
use crate::stream::{AllocatorType, ArrayType, RustStream, StreamData};

pub struct CrossbeamDevice {
  pub(crate) ncore: usize,
  pub(crate) pin: bool,
  pub(crate) core_ids: Vec<CoreId>,
}

impl CrossbeamDevice {
  pub fn new(ncore: usize, pin: bool) -> Self {
    let mut core_ids = match core_affinity::get_core_ids() {
      Some(xs) => xs,
      None => {
        colour::e_red_ln!("Cannot enumerate cores, pinning will not work if enabled");
        (0..ncore).map(|i| CoreId { id: i }).collect()
      }
    };
    core_ids.resize(ncore, core_ids[0]);
    CrossbeamDevice { ncore, pin, core_ids }
  }
}

impl CrossbeamDevice {
  // divide the length by the number of cores, the last core gets less work if it does not divide
  fn chunk_size(&self, len: usize) -> usize { (len as f64 / self.ncore as f64).ceil() as usize }

  // make a mutable chunk from the vec
  fn mk_mut_chunks<'a, T, A: AllocatorType>(&self, xs: &'a mut Vec<T, A>) -> ChunksMut<'a, T> {
    let len = xs.len();
    xs.chunks_mut(self.chunk_size(len))
  }

  // make a immutable chunk from the vec
  fn mk_chunks<'a, T, A: AllocatorType>(&self, xs: &'a mut Vec<T, A>) -> Chunks<'a, T> {
    xs.chunks(self.chunk_size(xs.len()))
  }
}

extern crate core_affinity;

// Crossbeam threaded version, it should be semantically equal to the single threaded version
impl<T: ArrayType + Sync + Send + Sum, A: AllocatorType + Sync + Send> RustStream<T>
  for StreamData<T, CrossbeamDevice, A>
{
  fn init_arrays(&mut self) {
    thread::scope(|s| {
      let init = self.init;
      let pin = self.device.pin;
      for (t, ((a, b), c)) in self.device.core_ids.iter().zip(
        self
          .device
          .mk_mut_chunks(&mut self.a)
          .zip(self.device.mk_mut_chunks(&mut self.b))
          .zip(self.device.mk_mut_chunks(&mut self.c)),
      ) {
        s.spawn(move |_| {
          if pin {
            core_affinity::set_for_current(*t);
          }
          for x in a.into_iter() {
            *x = init.0;
          }
          for x in b.into_iter() {
            *x = init.1;
          }
          for x in c.into_iter() {
            *x = init.2;
          }
        });
      }
    })
    .unwrap()
  }

  fn copy(&mut self) {
    thread::scope(|s| {
      let pin = self.device.pin;
      for (t, (c, a)) in self
        .device
        .core_ids
        .iter()
        .zip(self.device.mk_mut_chunks(&mut self.c).zip(self.device.mk_chunks(&mut self.a)))
      {
        s.spawn(move |_| {
          if pin {
            core_affinity::set_for_current(*t);
          }
          for i in 0..c.len() {
            c[i] = a[i];
          }
        });
      }
    })
    .unwrap()
  }

  fn mul(&mut self) {
    thread::scope(|s| {
      let pin = self.device.pin;
      let scalar = self.scalar;
      for (t, (b, c)) in self
        .device
        .core_ids
        .iter()
        .zip(self.device.mk_mut_chunks(&mut self.b).zip(self.device.mk_chunks(&mut self.c)))
      {
        s.spawn(move |_| {
          if pin {
            core_affinity::set_for_current(*t);
          }
          for i in 0..b.len() {
            b[i] = scalar * c[i];
          }
        });
      }
    })
    .unwrap()
  }

  fn add(&mut self) {
    thread::scope(|s| {
      let pin = self.device.pin;
      for (t, (c, (a, b))) in (&mut self.device.core_ids.iter()).zip(
        self
          .device
          .mk_mut_chunks(&mut self.c)
          .zip(self.device.mk_chunks(&mut self.a).zip(self.device.mk_chunks(&mut self.b))),
      ) {
        s.spawn(move |_| {
          if pin {
            core_affinity::set_for_current(*t);
          }
          for i in 0..c.len() {
            c[i] = a[i] + b[i];
          }
        });
      }
    })
    .unwrap()
  }

  fn triad(&mut self) {
    thread::scope(|s| {
      let pin = self.device.pin;
      let scalar = self.scalar;
      for (t, (a, (b, c))) in self.device.core_ids.iter().zip(
        self
          .device
          .mk_mut_chunks(&mut self.a)
          .zip(self.device.mk_chunks(&mut self.b).zip(self.device.mk_chunks(&mut self.c))),
      ) {
        s.spawn(move |_| {
          if pin {
            core_affinity::set_for_current(*t);
          }
          for i in 0..a.len() {
            a[i] = b[i] + scalar * c[i]
          }
        });
      }
    })
    .unwrap()
  }

  fn nstream(&mut self) {
    thread::scope(|s| {
      let pin = self.device.pin;
      let scalar = self.scalar;
      for (t, (a, (b, c))) in self.device.core_ids.iter().zip(
        self
          .device
          .mk_mut_chunks(&mut self.a)
          .zip(self.device.mk_chunks(&mut self.b).zip(self.device.mk_chunks(&mut self.c))),
      ) {
        s.spawn(move |_| {
          if pin {
            core_affinity::set_for_current(*t);
          }
          for i in 0..a.len() {
            a[i] += b[i] + scalar * c[i]
          }
        });
      }
    })
    .unwrap()
  }

  fn dot(&mut self) -> T {
    let mut partial_sum = vec![T::zero(); self.device.ncore];
    thread::scope(|s| {
      let pin = self.device.pin;
      let a = &self.a;
      let b = &self.b;
      let chunk_indices = |i: usize| {
        let chunk_size = self.device.chunk_size(self.size);
        let start = i * chunk_size;
        start..((start + chunk_size).min(self.size))
      };
      for (t, (n, acc)) in self.device.core_ids.iter().zip(partial_sum.iter_mut().enumerate()) {
        s.spawn(move |_| {
          if pin {
            core_affinity::set_for_current(*t);
          }
          let mut p = T::zero();
          for i in chunk_indices(n) {
            p += a[i] * b[i];
          }
          *acc = p;
        });
      }
    })
    .unwrap();
    partial_sum.into_iter().sum()
  }
}
