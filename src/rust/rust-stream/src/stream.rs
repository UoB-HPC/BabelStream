use num_traits::real::Real;
use num_traits::{NumAssign, Signed};
use std::alloc::Allocator;
use std::fmt::Debug;
use std::time::{Duration, Instant};

pub trait AllocatorType: Allocator + Copy + Clone + Default + Debug {}
impl<T: Allocator + Copy + Clone + Default + Debug> AllocatorType for T {}

pub struct StreamData<T, D, A: AllocatorType> {
  pub device: D,
  pub size: usize,
  pub scalar: T,
  pub init: (T, T, T),
  pub a: Vec<T, A>,
  pub b: Vec<T, A>,
  pub c: Vec<T, A>,
  pub needs_dealloc: bool,
}

#[inline(always)]
fn timed<F: FnOnce()>(f: F) -> Duration {
  let start = Instant::now();
  f();
  start.elapsed()
}

#[inline(always)]
fn timed_mut<T, F: FnMut() -> T>(f: &mut F) -> (Duration, T) {
  let start = Instant::now();
  let x = f();
  (start.elapsed(), x)
}

pub struct AllTiming<T> {
  pub copy: T,
  pub mul: T,
  pub add: T,
  pub triad: T,
  pub dot: T,
}

pub trait ArrayType: Real + NumAssign + Signed + Default + Debug {}
impl<T: Real + NumAssign + Signed + Default + Debug> ArrayType for T {}

impl<T: Default + Clone, D, A: AllocatorType> StreamData<T, D, A> {
  pub fn new_in(
    size: usize,
    scalar: T,
    init: (T, T, T),
    device: D,
    allocator: A,
    malloc: bool,     //
    initialise: bool, //
  ) -> StreamData<T, D, A> {
    let mk_vec = || {
      if malloc {
        extern crate libc;
        use std::mem;
        unsafe {
          // we do the typical C malloc with a NULL check here
          let bytes = mem::size_of::<T>() * size;
          let ptr = libc::malloc(bytes as libc::size_t) as *mut T;
          if ptr.is_null() {
            panic!(
              "Cannot allocate {} bytes in `sizeof(T) * size` (T = {}, size = {})",
              bytes,
              mem::size_of::<T>(),
              size
            );
          }
          let mut xs = Vec::from_raw_parts_in(ptr, size, size, allocator);
          if initialise {
            xs.fill(T::default());
          }
          xs
        }
      } else {
        if initialise {
          let mut xs = Vec::new_in(allocator);
          xs.resize(size, T::default());
          xs
        } else {
          // try not to touch the vec after allocation
          let mut xs = Vec::with_capacity_in(size, allocator);
          unsafe {
            xs.set_len(size);
          }
          xs
        }
      }
    };

    StreamData {
      device,
      size,
      scalar,
      init,
      a: mk_vec(),
      b: mk_vec(),
      c: mk_vec(),
      needs_dealloc: malloc,
    }
  }
  pub fn clean_up(self) {
    if self.needs_dealloc {
      unsafe {
        extern crate libc;
        let free_ts = move |xs: Vec<T, A>| {
          // make sure we don't call dealloc for vec anymore
          // XXX it's important we don't free xs.as_mut_ptr() here and use xs.into_raw_parts_with_alloc()
          // as that function handles drops semantic for us
          // if we free the the raw ptr directly, the compiler will still drop the vec and then segfault
          let (ptr, _, _, _) = xs.into_raw_parts_with_alloc();
          libc::free(ptr as *mut libc::c_void);
        };
        free_ts(self.a);
        free_ts(self.b);
        free_ts(self.c);
      }
    }
  }
}

pub trait RustStream<T: Default> {
  fn init_arrays(&mut self);
  fn read_arrays(&mut self) {} // default to no-op as most impl. doesn't need this
  fn copy(&mut self);
  fn mul(&mut self);
  fn add(&mut self);
  fn triad(&mut self);
  fn nstream(&mut self);
  fn dot(&mut self) -> T;

  fn run_init_arrays(&mut self) -> Duration {
    timed(|| {
      self.init_arrays();
    })
  }

  fn run_read_arrays(&mut self) -> Duration {
    timed(|| {
      self.read_arrays();
    })
  }

  fn run_all(&mut self, n: usize) -> (AllTiming<Vec<Duration>>, T) {
    let mut timings: AllTiming<Vec<Duration>> = AllTiming {
      copy: vec![Duration::default(); n],
      mul: vec![Duration::default(); n],
      add: vec![Duration::default(); n],
      triad: vec![Duration::default(); n],
      dot: vec![Duration::default(); n],
    };
    let mut last_sum = T::default();
    for i in 0..n {
      timings.copy[i] = timed(|| self.copy());
      timings.mul[i] = timed(|| self.mul());
      timings.add[i] = timed(|| self.add());
      timings.triad[i] = timed(|| self.triad());
      let (dot, sum) = timed_mut(&mut || self.dot());
      timings.dot[i] = dot;
      last_sum = sum;
    }
    (timings, last_sum)
  }

  fn run_triad(&mut self, n: usize) -> Duration {
    timed(|| {
      for _ in 0..n {
        self.triad();
      }
    })
  }

  fn run_nstream(&mut self, n: usize) -> Vec<Duration> {
    (0..n).map(|_| timed(|| self.nstream())).collect::<Vec<_>>()
  }
}
