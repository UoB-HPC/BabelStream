use crate::stream::{AllocatorType, ArrayType, RustStream, StreamData};
use core_affinity::CoreId;

pub struct SerialDevice {
  pub(crate) pin: bool,
}

// single threaded version
impl<T: ArrayType, A: AllocatorType> RustStream<T> for StreamData<T, SerialDevice, A> {
  fn init_arrays(&mut self) {
    if self.device.pin {
      core_affinity::set_for_current(
        match core_affinity::get_core_ids().as_ref().map(|x| x.first()) {
          Some(Some(x)) => *x,
          _ => CoreId { id: 0 },
        },
      );
    }
    self.a.fill(self.init.0);
    self.b.fill(self.init.1);
    self.c.fill(self.init.2);
  }

  fn copy(&mut self) {
    for i in 0..self.size {
      self.c[i] = self.a[i];
    }
  }

  fn mul(&mut self) {
    for i in 0..self.size {
      self.b[i] = self.scalar * self.c[i];
    }
  }

  fn add(&mut self) {
    for i in 0..self.size {
      self.c[i] = self.a[i] + self.b[i];
    }
  }

  fn triad(&mut self) {
    for i in 0..self.size {
      self.a[i] = self.b[i] + self.scalar * self.c[i];
    }
  }

  fn nstream(&mut self) {
    for i in 0..self.size {
      self.a[i] += self.b[i] + self.scalar * self.c[i];
    }
  }

  fn dot(&mut self) -> T {
    let mut sum = T::default();
    for i in 0..self.size {
      sum += self.a[i] * self.b[i];
    }
    sum
  }
}
