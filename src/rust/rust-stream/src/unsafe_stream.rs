extern crate core_affinity;

use std::alloc::Allocator;
use std::iter::Sum;
use std::ops::Range;

use crate::stream::{AllocatorType, ArrayType, RustStream, StreamData};

use self::core_affinity::CoreId;

#[derive(Debug, Copy, Clone)]
struct UnsafeData<T>(*mut T, usize);

impl<T: ArrayType> UnsafeData<T> {
  fn empty() -> UnsafeData<T> { UnsafeData(([] as [T; 0]).as_mut_ptr(), 0) }
  fn new<A: Allocator>(xs: &mut Vec<T, A>) -> UnsafeData<T> {
    UnsafeData(xs.as_mut_ptr(), xs.len())
  }

  fn get_slice(&self) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.0, self.1) } }
}

unsafe impl<T> Send for UnsafeData<T> {}
unsafe impl<T> Sync for UnsafeData<T> {}

#[derive(Debug, Copy, Clone)]
struct UnsafeRefs<T> {
  a: UnsafeData<T>,
  b: UnsafeData<T>,
  c: UnsafeData<T>,
}

unsafe impl<T> Send for UnsafeRefs<T> {}
unsafe impl<T> Sync for UnsafeRefs<T> {}

pub struct UnsafeDevice<T: ArrayType> {
  pub(crate) ncore: usize,
  pub(crate) pin: bool,
  pub(crate) core_ids: Vec<CoreId>,
  data: UnsafeRefs<T>,
}

impl<T: ArrayType> UnsafeDevice<T> {
  pub fn new(ncore: usize, pin: bool) -> Self {
    let mut core_ids = match core_affinity::get_core_ids() {
      Some(xs) => xs,
      None => {
        colour::e_red_ln!("Cannot enumerate cores, pinning will not work if enabled");
        (0..ncore).map(|i| CoreId { id: i }).collect()
      }
    };
    core_ids.resize(ncore, core_ids[0]);

    UnsafeDevice {
      ncore,
      pin,
      core_ids,
      data: UnsafeRefs { a: UnsafeData::empty(), b: UnsafeData::empty(), c: UnsafeData::empty() },
    }
  }

  fn thread_ranges(&self, len: usize) -> Vec<(usize, Range<usize>)> {
    let chunk = (len as f64 / self.ncore as f64).ceil() as usize;
    (0..self.ncore)
      .map(|t| {
        (t, if t == self.ncore - 1 { (t * chunk)..len } else { (t * chunk)..((t + 1) * chunk) })
      })
      .collect::<Vec<_>>()
  }
}

// Unsafe threaded version, it should be semantically equal to the single threaded version
impl<T: 'static + ArrayType + Sync + Send + Sum, A: AllocatorType + Sync + Send> RustStream<T>
  for StreamData<T, UnsafeDevice<T>, A>
{
  fn init_arrays(&mut self) {
    self.device.data.a = UnsafeData::new(&mut self.a);
    self.device.data.b = UnsafeData::new(&mut self.b);
    self.device.data.c = UnsafeData::new(&mut self.c);
    let init = self.init;
    let pin = self.device.pin;
    let data = self.device.data;
    self
      .device
      .thread_ranges(self.size)
      .into_iter()
      .map(|(t, r)| {
        let core = self.device.core_ids[t];
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = data.a.get_slice();
          let b = data.b.get_slice();
          let c = data.c.get_slice();
          for i in r {
            a[i] = init.0;
            b[i] = init.1;
            c[i] = init.2;
          }
        })
      })
      .collect::<Vec<_>>()
      .into_iter()
      .for_each(|t| t.join().unwrap());
  }

  fn copy(&mut self) {
    let pin = self.device.pin;
    let data = self.device.data;
    self
      .device
      .thread_ranges(self.size)
      .into_iter()
      .map(|(t, r)| {
        let core = self.device.core_ids[t];
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = data.a.get_slice();
          let c = data.c.get_slice();
          for i in r {
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
    let data = self.device.data;
    self
      .device
      .thread_ranges(self.size)
      .into_iter()
      .map(|(t, r)| {
        let core = self.device.core_ids[t];
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let b = data.b.get_slice();
          let c = data.c.get_slice();
          for i in r {
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
    let data = self.device.data;
    self
      .device
      .thread_ranges(self.size)
      .into_iter()
      .map(|(t, r)| {
        let core = self.device.core_ids[t];
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = data.a.get_slice();
          let b = data.b.get_slice();
          let c = data.c.get_slice();
          for i in r {
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
    let data = self.device.data;
    self
      .device
      .thread_ranges(self.size)
      .into_iter()
      .map(|(t, r)| {
        let core = self.device.core_ids[t];
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = data.a.get_slice();
          let b = data.b.get_slice();
          let c = data.c.get_slice();
          for i in r {
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
    let data = self.device.data;
    self
      .device
      .thread_ranges(self.size)
      .into_iter()
      .map(|(t, r)| {
        let core = self.device.core_ids[t];
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = data.a.get_slice();
          let b = data.b.get_slice();
          let c = data.c.get_slice();
          for i in r {
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
    let data = self.device.data;
    self
      .device
      .thread_ranges(self.size)
      .into_iter()
      .map(|(t, r)| {
        let core = self.device.core_ids[t];
        std::thread::spawn(move || {
          if pin {
            core_affinity::set_for_current(core);
          }
          let a = data.a.get_slice();
          let b = data.b.get_slice();
          let mut p = T::default();
          for i in r {
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
