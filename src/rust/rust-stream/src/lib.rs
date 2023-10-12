#![feature(allocator_api)]
#![feature(vec_into_raw_parts)]

use std::alloc::System;
use std::env;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::mem::size_of;
use std::time::Duration;

use num_traits::abs;
use structopt::StructOpt;
use tabular::{Row, Table};

use crate::arc_stream::ArcDevice;
use crate::crossbeam_stream::CrossbeamDevice;
use crate::plain_stream::SerialDevice;
use crate::rayon_stream::RayonDevice;
use crate::stream::{AllocatorType, ArrayType, RustStream, StreamData};
use crate::unsafe_stream::UnsafeDevice;

mod arc_stream;
mod crossbeam_stream;
mod plain_stream;
mod rayon_stream;
mod stream;
mod unsafe_stream;

#[derive(Debug, StructOpt)]
struct Options {
  /// List available devices
  #[structopt(long)]
  list: bool,
  /// Select device at <device>
  #[structopt(long, default_value = "0")]
  device: usize,
  /// Run the test <numtimes> times (NUM >= 2)
  #[structopt(long, short = "n", default_value = "100")]
  numtimes: usize,
  /// Use <arraysize> elements in the array
  #[structopt(long, short = "s", default_value = "33554432")]
  arraysize: usize,
  /// Use floats (rather than doubles)
  #[structopt(long)]
  float: bool,
  /// Only run triad
  #[structopt(long)]
  triad_only: bool,
  /// Only run nstream
  #[structopt(long)]
  nstream_only: bool,
  /// Output as csv table
  #[structopt(long)]
  csv: bool,
  /// Use MiB=2^20 for bandwidth calculation (default MB=10^6)
  #[structopt(long)]
  mibibytes: bool,
  /// Use libc malloc instead of the Rust's allocator for benchmark array allocation
  #[structopt(name = "malloc", long)]
  malloc: bool,
  /// Initialise each benchmark array at allocation time on the main thread
  #[structopt(name = "init", long)]
  init: bool,
  /// Pin threads to distinct cores, this has NO effect in Rayon devices
  #[structopt(long)]
  pin: bool,
}

#[derive(PartialEq)]
enum Benchmark {
  All,
  Triad,
  NStream,
}

fn check_solution<T: ArrayType + Display + Sum + Into<f64>, D, A: AllocatorType>(
  benchmark: Benchmark, numtimes: usize, vec: &StreamData<T, D, A>, dot_sum: Option<T>,
) -> bool {
  let (mut gold_a, mut gold_b, mut gold_c) = vec.init;
  for _ in 0..numtimes {
    match benchmark {
      Benchmark::All => {
        gold_c = gold_a;
        gold_b = vec.scalar * gold_c;
        gold_c = gold_a + gold_b;
        gold_a = gold_b + vec.scalar * gold_c;
      }
      Benchmark::Triad => {
        gold_a = gold_b + vec.scalar * gold_c;
      }
      Benchmark::NStream => {
        gold_a += gold_b + vec.scalar * gold_c;
      }
    };
  }
  let tolerance = T::epsilon().into() * 100.0f64;
  let validate_xs = |name: &str, xs: &Vec<T, A>, from: T| {
    let error = (xs.iter().map(|x| abs(*x - from)).sum::<T>()).into() / xs.len() as f64;
    let fail = error > tolerance;
    if fail {
      eprintln!("Validation failed on {}[]. Average error {} ", name, error);
    }
    !fail
  };
  let a_ok = validate_xs("a", &vec.a, gold_a);
  let b_ok = validate_xs("b", &vec.b, gold_b);
  let c_ok = validate_xs("c", &vec.c, gold_c);
  let dot_ok = dot_sum.map_or(true, |sum| {
    let gold_sum = (gold_a * gold_b).into() * vec.size as f64;
    let error = abs((sum.into() - gold_sum) / gold_sum);
    let fail = error > 1.0e-8;
    if fail {
      eprintln!(
        "Validation failed on sum. Error {} \nSum was {} but should be {}",
        error, sum, gold_sum
      );
    }
    !fail
  });

  a_ok && b_ok && c_ok && dot_ok
}

fn run_cpu<T: ArrayType + Sync + Send + Sum + Into<f64> + Display, D, A: AllocatorType>(
  option: &Options, mut stream: StreamData<T, D, A>,
) -> bool
where StreamData<T, D, A>: RustStream<T> {
  let benchmark = match (option.nstream_only, option.triad_only) {
    (true, false) => Benchmark::NStream,
    (false, true) => Benchmark::Triad,
    (false, false) => Benchmark::All,
    (true, true) => {
      panic!("Both triad and nstream are enabled, pick one or omit both to run all benchmarks")
    }
  };

  let array_bytes = option.arraysize * size_of::<T>();
  let total_bytes = array_bytes * 3;
  let (mega_scale, mega_suffix, giga_scale, giga_suffix) = if !option.mibibytes {
    (1.0e-6, "MB", 1.0e-9, "GB")
  } else {
    (2f64.powi(-20), "MiB", 2f64.powi(-30), "GiB")
  };

  if !option.csv {
    println!(
      "Running {} {} times",
      match benchmark {
        Benchmark::All => "kernels",
        Benchmark::Triad => "triad",
        Benchmark::NStream => "nstream",
      },
      option.numtimes
    );

    if benchmark == Benchmark::Triad {
      println!("Number of elements: {}", option.arraysize);
    }

    println!("Precision: {}", if option.float { "float" } else { "double" });
    println!(
      "Array size: {:.1} {} (={:.1} {})",
      mega_scale * array_bytes as f64,
      mega_suffix,
      giga_scale * array_bytes as f64,
      giga_suffix
    );
    println!(
      "Total size: {:.1} {} (={:.1} {})",
      mega_scale * total_bytes as f64,
      mega_suffix,
      giga_scale * total_bytes as f64,
      giga_suffix
    );
  }

  let init = stream.run_init_arrays();

  let tabulate = |xs: &Vec<Duration>, name: &str, t_size: usize| -> Vec<(&str, String)> {
    let tail = &xs[1..]; // tail only

    // do stats
    let max = tail.iter().max().map(|d| d.as_secs_f64());
    let min = tail.iter().min().map(|d| d.as_secs_f64());
    match (min, max) {
      (Some(min), Some(max)) => {
        let avg: f64 = tail.iter().map(|d| d.as_secs_f64()).sum::<f64>() / tail.len() as f64;
        let mbps = mega_scale * (t_size as f64) / min;
        if option.csv {
          vec![
            ("function", name.to_string()),
            ("num_times", option.numtimes.to_string()),
            ("n_elements", option.arraysize.to_string()),
            ("sizeof", t_size.to_string()),
            (
              if option.mibibytes { "max_mibytes_per_sec" } else { "max_mbytes_per_sec" },
              mbps.to_string(),
            ),
            ("min_runtime", min.to_string()),
            ("max_runtime", max.to_string()),
            ("avg_runtime", avg.to_string()),
          ]
        } else {
          vec![
            ("Function", name.to_string()),
            (if option.mibibytes { "MiBytes/sec" } else { "MBytes/sec" }, format!("{:.3}", mbps)),
            ("Min (sec)", format!("{:.5}", min)),
            ("Max", format!("{:.5}", max)),
            ("Average", format!("{:.5}", avg)),
          ]
        }
      }
      (_, _) => panic!("No min/max element for {}(size={})", name, t_size),
    }
  };

  let tabulate_all = |xs: Vec<Vec<(&str, String)>>| {
    match xs.as_slice() {
      [head, ..] => {
        if option.csv {
          println!("{}", head.iter().map(|(col, _)| *col).collect::<Vec<_>>().join(","));
          for kvs in xs {
            println!("{}", kvs.iter().map(|(_, val)| val.clone()).collect::<Vec<_>>().join(","));
          }
        } else {
          let mut table = Table::new(&vec!["{:<}"; head.len()].join("    "));
          table.add_row(head.iter().fold(Row::new(), |row, (col, _)| row.with_cell(col)));
          for kvs in xs {
            table.add_row(kvs.iter().fold(Row::new(), |row, (_, val)| row.with_cell(val)));
          }
          print!("{}", table);
        }
      }
      _ => panic!("Empty tabulation"),
    };
  };

  let show_setup = |init: Duration, read: Duration| {
    let setup = vec![
      ("Init", init.as_secs_f64(), 3 * array_bytes),
      ("Read", read.as_secs_f64(), 3 * array_bytes),
    ];
    if option.csv {
      tabulate_all(
        setup
          .iter()
          .map(|(name, elapsed, t_size)| {
            vec![
              ("phase", name.to_string()),
              ("n_elements", option.arraysize.to_string()),
              ("sizeof", t_size.to_string()),
              (
                if option.mibibytes { "max_mibytes_per_sec" } else { "max_mbytes_per_sec" },
                (mega_scale * (*t_size as f64) / elapsed).to_string(),
              ),
              ("runtime", elapsed.to_string()),
            ]
          })
          .collect::<Vec<_>>(),
      );
    } else {
      for (name, elapsed, t_size) in setup {
        println!(
          "{}: {:.5} s (={:.5} {})",
          name,
          elapsed,
          mega_scale * (t_size as f64) / elapsed,
          if option.mibibytes { "MiBytes/sec" } else { "MBytes/sec" }
        );
      }
    }
  };

  let solutions_correct = match benchmark {
    Benchmark::All => {
      let (results, sum) = stream.run_all(option.numtimes);
      let read = stream.run_read_arrays();
      show_setup(init, read);
      let correct = check_solution(benchmark, option.numtimes, &stream, Some(sum));
      tabulate_all(vec![
        tabulate(&results.copy, "Copy", 2 * array_bytes),
        tabulate(&results.mul, "Mul", 2 * array_bytes),
        tabulate(&results.add, "Add", 3 * array_bytes),
        tabulate(&results.triad, "Triad", 3 * array_bytes),
        tabulate(&results.dot, "Dot", 2 * array_bytes),
      ]);
      correct
    }
    Benchmark::NStream => {
      let results = stream.run_nstream(option.numtimes);
      let read = stream.run_read_arrays();
      show_setup(init, read);
      let correct = check_solution(benchmark, option.numtimes, &stream, None);
      tabulate_all(vec![tabulate(&results, "Nstream", 4 * array_bytes)]);
      correct
    }
    Benchmark::Triad => {
      let results = stream.run_triad(option.numtimes);
      let read = stream.run_read_arrays();
      show_setup(init, read);
      let correct = check_solution(benchmark, option.numtimes, &stream, None);
      let total_bytes = 3 * array_bytes * option.numtimes;
      let bandwidth = giga_scale * (total_bytes as f64 / results.as_secs_f64());
      println!("Runtime (seconds): {:.5}", results.as_secs_f64());
      println!("Bandwidth ({}/s): {:.3} ", giga_suffix, bandwidth);
      correct
    }
  };
  stream.clean_up();
  solutions_correct
}

const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

static START_A: f32 = 0.1;
static START_B: f32 = 0.2;
static START_C: f32 = 0.0;
static START_SCALAR: f32 = 0.4;

static FLOAT_INIT_SCALAR: f32 = START_SCALAR;
static FLOAT_INIT: (f32, f32, f32) = (START_A, START_B, START_C);

static DOUBLE_INIT_SCALAR: f64 = START_SCALAR as f64;
static DOUBLE_INIT: (f64, f64, f64) = (START_A as f64, START_B as f64, START_C as f64);

pub fn run(args: &Vec<String>) -> bool {
  let opt: Options = Options::from_iter(args);

  if opt.numtimes < 2 {
    panic!("numtimes must be >= 2")
  }

  let alloc = System;
  let alloc_name = if opt.malloc { "libc-malloc" } else { "rust-system" };

  fn mk_data<T: ArrayType, D, A: AllocatorType>(
    opt: &Options, init: (T, T, T), scalar: T, dev: D, alloc: A,
  ) -> StreamData<T, D, A> {
    StreamData::new_in(opt.arraysize, scalar, init, dev, alloc, opt.malloc, opt.init)
  }

  let num_thread_key = "BABELSTREAM_NUM_THREADS";
  let max_ncores = num_cpus::get();
  let ncores = match env::var(num_thread_key) {
    Ok(v) => match v.parse::<i64>() {
      Err(bad) => {
        colour::e_yellow_ln!(
          "Cannot parse {} (reason: {}), defaulting to {}",
          bad,
          num_thread_key,
          max_ncores
        );
        max_ncores
      }
      Ok(n) if n <= 0 || n > max_ncores as i64 => {
        println!("{} out of bound ({}), defaulting to {}", num_thread_key, n, max_ncores);
        max_ncores
      }
      Ok(n) => n as usize,
    },
    Err(_) => {
      println!("{} not set, defaulting to max ({})", num_thread_key, max_ncores);
      max_ncores
    }
  };

  let rayon_device = &|| {
    let rayon_num_thread_key = "RAYON_NUM_THREADS";
    if env::var(rayon_num_thread_key).is_ok() {
      colour::e_yellow_ln!("{} is ignored, set {} instead", rayon_num_thread_key, num_thread_key)
    }
    let dev = RayonDevice {
      pool: rayon::ThreadPoolBuilder::default().num_threads(ncores).build().unwrap(),
    };
    if !opt.csv {
      println!("Using {} thread(s), alloc={}", dev.pool.current_num_threads(), alloc_name);
      if opt.pin {
        colour::e_yellow_ln!("Pinning threads have no effect on Rayon!")
      }
    }
    if opt.float {
      run_cpu(&opt, mk_data(&opt, FLOAT_INIT, FLOAT_INIT_SCALAR, dev, alloc))
    } else {
      run_cpu(&opt, mk_data(&opt, DOUBLE_INIT, DOUBLE_INIT_SCALAR, dev, alloc))
    }
  };

  let arc_device = &|| {
    if !opt.csv {
      println!("Using {} thread, pin={}, alloc={}", ncores, opt.pin, alloc_name);
    }
    if opt.float {
      let dev = ArcDevice::<f32, _>::new(ncores, opt.pin, alloc);
      run_cpu(&opt, mk_data(&opt, FLOAT_INIT, FLOAT_INIT_SCALAR, dev, alloc))
    } else {
      let dev = ArcDevice::<f64, _>::new(ncores, opt.pin, alloc);
      run_cpu(&opt, mk_data(&opt, DOUBLE_INIT, DOUBLE_INIT_SCALAR, dev, alloc))
    }
  };

  let unsafe_device = &|| {
    if !opt.csv {
      println!("Using {} thread, pin={}, alloc={}", ncores, opt.pin, alloc_name);
    }
    if opt.float {
      let dev = UnsafeDevice::<f32>::new(ncores, opt.pin);
      run_cpu(&opt, mk_data(&opt, FLOAT_INIT, FLOAT_INIT_SCALAR, dev, alloc))
    } else {
      let dev = UnsafeDevice::<f64>::new(ncores, opt.pin);
      run_cpu(&opt, mk_data(&opt, DOUBLE_INIT, DOUBLE_INIT_SCALAR, dev, alloc))
    }
  };

  let crossbeam_device = &|| {
    let dev = CrossbeamDevice::new(ncores, opt.pin);
    if !opt.csv {
      println!("Using {} thread(s), pin={}, alloc={}", ncores, opt.pin, alloc_name)
    }
    if opt.float {
      run_cpu(&opt, mk_data(&opt, FLOAT_INIT, FLOAT_INIT_SCALAR, dev, alloc))
    } else {
      run_cpu(&opt, mk_data(&opt, DOUBLE_INIT, DOUBLE_INIT_SCALAR, dev, alloc))
    }
  };

  let st_device = &|| {
    let dev = SerialDevice { pin: opt.pin };
    if !opt.csv {
      println!("Using 1 thread, pin={}, alloc={}", opt.pin, alloc_name);
    }
    if opt.float {
      run_cpu(&opt, mk_data(&opt, FLOAT_INIT, FLOAT_INIT_SCALAR, dev, alloc))
    } else {
      run_cpu(&opt, mk_data(&opt, DOUBLE_INIT, DOUBLE_INIT_SCALAR, dev, alloc))
    }
  };

  let devices: Vec<(String, &'_ dyn Fn() -> bool)> = vec![
    ("CPU (Single threaded)".to_string(), st_device),
    ("CPU (Rayon)".to_string(), rayon_device),
    (format!("CPU (Arc, pinning={})", opt.pin), arc_device),
    (format!("CPU (Unsafe, pinning={})", opt.pin), unsafe_device),
    (format!("CPU (Crossbeam, pinning={})", opt.pin), crossbeam_device),
  ];

  if opt.list {
    devices.iter().enumerate().for_each(|(i, (name, _))| {
      println!("[{}] {}", i, name);
    });
    true
  } else {
    match devices.get(opt.device) {
      Some((name, run)) => {
        if !&opt.csv {
          println!(
            "BabelStream\n\
                              Version: {}\n\
                              Implementation: Rust; {}",
            VERSION.unwrap_or("unknown"),
            name
          );
          if opt.init {
            println!("Initialising arrays on main thread");
          }
        }
        run()
      }
      None => {
        eprintln!("Device index {} not available", opt.device);
        false
      }
    }
  }
}
