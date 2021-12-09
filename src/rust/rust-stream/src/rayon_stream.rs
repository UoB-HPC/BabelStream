use std::iter::Sum;

use rayon::prelude::*;
use rayon::ThreadPool;

use crate::stream::{AllocatorType, ArrayType, RustStream, StreamData};

pub struct RayonDevice {
  pub(crate) pool: ThreadPool,
}

//  Rayon version, it should be semantically equal to the single threaded version
impl<T: ArrayType + Sync + Send + Sum, A: AllocatorType + Sync + Send> RustStream<T>
  for StreamData<T, RayonDevice, A>
{
  fn init_arrays(&mut self) {
    let init = self.init;
    self.a.par_iter_mut().for_each(|v| *v = init.0);
    self.b.par_iter_mut().for_each(|v| *v = init.1);
    self.c.par_iter_mut().for_each(|v| *v = init.2);
  }

  fn copy(&mut self) {
    let a = &self.a;
    let c = &mut self.c;
    self.device.pool.install(|| {
      (*c).par_iter_mut().enumerate().for_each(|(i, c)| *c = a[i]);
    });
  }

  fn mul(&mut self) {
    let scalar = self.scalar;
    let c = &self.c;
    let b = &mut self.b;
    self
      .device
      .pool
      .install(|| (*b).par_iter_mut().enumerate().for_each(|(i, b)| *b = scalar * c[i]));
  }

  fn add(&mut self) {
    let a = &self.a;
    let b = &self.b;
    let c = &mut self.c;
    self.device.pool.install(|| (*c).par_iter_mut().enumerate().for_each(|(i, c)| *c = a[i] + b[i]))
  }

  fn triad(&mut self) {
    let scalar = self.scalar;
    let a = &mut self.a;
    let b = &self.b;
    let c = &self.c;
    self
      .device
      .pool
      .install(|| (*a).par_iter_mut().enumerate().for_each(|(i, a)| *a = b[i] + scalar * c[i]))
  }

  fn nstream(&mut self) {
    let scalar = self.scalar;
    let a = &mut self.a;
    let b = &self.b;
    let c = &self.c;
    self
      .device
      .pool
      .install(|| (*a).par_iter_mut().enumerate().for_each(|(i, a)| *a += b[i] + scalar * c[i]))
  }

  fn dot(&mut self) -> T {
    let a = &self.a;
    let b = &self.b;
    self.device.pool.install(|| {
      (0..self.size).into_par_iter().fold(|| T::default(), |acc, i| acc + a[i] * b[i]).sum::<T>()
    })
  }
}
