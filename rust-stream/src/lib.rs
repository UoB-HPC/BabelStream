#![feature(allocator_api)]
#![feature(vec_into_raw_parts)]

use std::alloc::System;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::mem::size_of;
use std::time::Duration;

use num_traits::abs;
use structopt::StructOpt;
use tabular::{Row, Table};

use crate::crossbeam_stream::ThreadedDevice;
use crate::plain_stream::SerialDevice;
use crate::rayon_stream::RayonDevice;
use crate::stream::{AllocatorType, ArrayType, RustStream, StreamData};

mod crossbeam_stream;
mod plain_stream;
mod rayon_stream;
mod stream;

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
where
  StreamData<T, D, A>: RustStream<T>,
{
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

  stream.init_arrays();

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

  let solutions_correct = match benchmark {
    Benchmark::All => {
      let (results, sum) = stream.run_all(option.numtimes);
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
      let correct = check_solution(benchmark, option.numtimes, &stream, None);
      tabulate_all(vec![tabulate(&results, "Nstream", 4 * array_bytes)]);
      correct
    }
    Benchmark::Triad => {
      let results = stream.run_triad(option.numtimes);
      let correct = check_solution(benchmark, option.numtimes, &stream, None);
      let total_bytes = 3 * array_bytes * option.numtimes;
      let bandwidth = giga_scale * (total_bytes as f64 / results.as_secs_f64());
      println!("Runtime (seconds): {:.5}", results.as_secs_f64());
      println!("Bandwidth ({}/s): {:.3} ", giga_suffix, bandwidth);
      correct
    }
  };
  &stream.clean_up();
  solutions_correct
}

const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

static START_A: f32 = 0.1;
static START_B: f32 = 0.2;
static START_C: f32 = 0.0;
static START_SCALAR: f32 = 0.4;

static FLOAT_INIT_SCALAR: f32 = START_SCALAR;
static FLOAT_INIT: (f32, f32, f32) = (START_A, START_B, START_C);

static DOUBLE_START_SCALAR: f64 = START_SCALAR as f64;
static DOUBLE_INIT: (f64, f64, f64) = (START_A as f64, START_B as f64, START_C as f64);

pub fn run(args: &Vec<String>) -> bool {

  let options: Options = Options::from_iter(args);

  if options.numtimes < 2 {
    panic!("numtimes must be >= 2")
  }

  let alloc = System;
  let alloc_name = if options.malloc { "libc-malloc" } else { "rust-system" };

  let rayon_device = &|| {
    let dev = RayonDevice { pool: rayon::ThreadPoolBuilder::default().build().unwrap() };
    if !options.csv {
      println!("Using {} thread(s), alloc={}", dev.pool.current_num_threads(), alloc_name);
      if options.pin {
        colour::e_yellow_ln!("Pinning threads have no effect on Rayon!")
      }
    }
    if options.float {
      run_cpu(
        &options,
        StreamData::new_in(
          options.arraysize,
          FLOAT_INIT_SCALAR,
          FLOAT_INIT,
          dev,
          alloc,
          options.malloc,
          options.init,
        ),
      )
    } else {
      run_cpu(
        &options,
        StreamData::new_in(
          options.arraysize,
          DOUBLE_START_SCALAR,
          DOUBLE_INIT,
          dev,
          alloc,
          options.malloc,
          options.init,
        ),
      )
    }
  };

  let crossbeam_device = &|| {
    let ncores = num_cpus::get();
    let dev = ThreadedDevice::new(ncores, options.pin);
    if !options.csv {
      println!("Using {} thread(s), pin={}, alloc={}", ncores, options.pin, alloc_name)
    }
    if options.float {
      run_cpu(
        &options,
        StreamData::new_in(
          options.arraysize,
          FLOAT_INIT_SCALAR,
          FLOAT_INIT,
          dev,
          alloc,
          options.malloc,
          options.init,
        ),
      )
    } else {
      run_cpu(
        &options,
        StreamData::new_in(
          options.arraysize,
          DOUBLE_START_SCALAR,
          DOUBLE_INIT,
          dev,
          alloc,
          options.malloc,
          options.init,
        ),
      )
    }
  };
  let st_device = &|| {
    let dev = SerialDevice { pin: options.pin };
    if !options.csv {
      println!("Using 1 thread, pin={}, alloc={}", options.pin, alloc_name);
    }
    if options.float {
      run_cpu(
        &options,
        StreamData::new_in(
          options.arraysize,
          FLOAT_INIT_SCALAR,
          FLOAT_INIT,
          dev,
          alloc,
          options.malloc,
          options.init,
        ),
      )
    } else {
      run_cpu(
        &options,
        StreamData::new_in(
          options.arraysize,
          DOUBLE_START_SCALAR,
          DOUBLE_INIT,
          dev,
          alloc,
          options.malloc,
          options.init,
        ),
      )
    }
  };
  let devices: Vec<(String, &'_ dyn Fn() -> bool)> = vec![
    ("CPU (Rayon)".to_string(), rayon_device),
    (format!("CPU (Crossbeam, pinning={})", options.pin), crossbeam_device),
    ("CPU (Single threaded)".to_string(), st_device),
  ];

  if options.list {
    devices.iter().enumerate().for_each(|(i, (name, _))| {
      println!("[{}] {}", i, name);
    });
    true
  } else {
    match devices.get(options.device) {
      Some((name, run)) => {
        if !&options.csv {
          println!(
            "BabelStream\n\
                              Version: {}\n\
                              Implementation: Rust; {}",
            VERSION.unwrap_or("unknown"),
            name
          );
          if options.init {
            println!("Initialising arrays on main thread");
          }
        }
        run()
      }
      None => {
        eprintln!("Device index {} not available", options.device);
        false
      }
    }
  }
}
