use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::mem::size_of;
use std::time::{Duration, Instant};

use num_traits::{abs, NumAssign, Signed};
use num_traits::real::Real;
use rayon::prelude::*;
use structopt::StructOpt;
use tabular::{Row, Table};

#[derive(Debug, StructOpt)]
struct Options {
    /// List available devices
    #[structopt(long)] list: bool,
    /// Select device at <device>
    #[structopt(long, default_value = "0")] device: usize,
    /// Run the test <numtimes> times (NUM >= 2)
    #[structopt(long, default_value = "100")] numtimes: usize,
    /// Use <arraysize> elements in the array
    #[structopt(long, default_value = "33554432")] arraysize: usize,
    /// Use floats (rather than doubles)
    #[structopt(long)] float: bool,
    /// Only run triad
    #[structopt(long)] triad_only: bool,
    /// Only run nstream
    #[structopt(long)] nstream_only: bool,
    /// Output as csv table
    #[structopt(long)] csv: bool,
    /// Use MiB=2^20 for bandwidth calculation (default MB=10^6)
    #[structopt(long)] mibibytes: bool,
}

#[derive(PartialEq)]
enum Benchmark { All, Triad, NStream }

struct StreamData<T> {
    size: usize,
    scalar: T,
    a: Vec<T>,
    b: Vec<T>,
    c: Vec<T>,
}

impl<T: Default + Clone> StreamData<T> {
    pub fn new(size: usize, scalar: T) -> StreamData<T> {
        StreamData {
            size,
            scalar,
            a: vec![T::default(); size],
            b: vec![T::default(); size],
            c: vec![T::default(); size],
        }
    }
}

struct PlainFor;

struct RayonPar;

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

struct AllTiming<T> { copy: T, mul: T, add: T, triad: T, dot: T }

trait RustStream<T: Default, K> {
    fn init_arrays(&mut self, init: (T, T, T));
    fn copy(&mut self);
    fn mul(&mut self);
    fn add(&mut self);
    fn triad(&mut self);
    fn nstream(&mut self);
    fn dot(&mut self) -> T;

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
        timed(|| for _ in 0..n { self.triad(); })
    }

    fn run_nstream(&mut self, n: usize) -> Vec<Duration> {
        (0..n).map(|_| timed(|| self.nstream())).collect::<Vec<_>>()
    }
}

trait ArrayType: Real + NumAssign + Signed + Default {}

impl<T: Real + NumAssign + Signed + Default> ArrayType for T {}

// single threaded version
impl<T: ArrayType> RustStream<T, PlainFor> for StreamData<T> {
    fn init_arrays(&mut self, init: (T, T, T)) {
        self.a.fill(init.0);
        self.b.fill(init.1);
        self.c.fill(init.2);
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
            self.a[i] += self.b[i] * self.scalar * self.c[i];
        }
    }

    fn dot(&mut self) -> T {
        let mut sum: T = T::default();
        for i in 0..self.size {
            sum += self.a[i] * self.b[i];
        }
        sum
    }
}

//  Rayon version, it should be semantically equal to the single threaded version
impl<T: ArrayType + Sync + Send + Sum> RustStream<T, RayonPar> for StreamData<T> {
    fn init_arrays(&mut self, init: (T, T, T)) {
        self.a.fill(init.0);
        self.b.fill(init.1);
        self.c.fill(init.2);
    }

    fn copy(&mut self) {
        let a = &self.a;
        self.c.par_iter_mut().enumerate().for_each(|(i, c)| *c = a[i])
    }

    fn mul(&mut self) {
        let c = &self.c;
        let scalar = &self.scalar;
        self.b.par_iter_mut().enumerate().for_each(|(i, b)| *b = *scalar * c[i])
    }

    fn add(&mut self) {
        let a = &self.a;
        let b = &self.b;
        self.c.par_iter_mut().enumerate().for_each(|(i, c)| *c = a[i] + b[i])
    }

    fn triad(&mut self) {
        let scalar = &self.scalar;
        let b = &self.b;
        let c = &self.c;
        self.a.par_iter_mut().enumerate().for_each(|(i, a)| *a = b[i] + *scalar * c[i])
    }

    fn nstream(&mut self) {
        let scalar = &self.scalar;
        let b = &self.b;
        let c = &self.c;
        self.a.par_iter_mut().enumerate().for_each(|(i, a)| *a += b[i] + *scalar * c[i])
    }

    fn dot(&mut self) -> T {
        let a = &self.a;
        let b = &self.b;
        (0..self.size).into_par_iter().fold(|| T::default(), |acc, i| acc + a[i] * b[i]).sum::<T>()
    }
}

fn validate<T: ArrayType + Display + Sum + Into<f64>>(
    benchmark: Benchmark,
    numtimes: usize,
    vec: &StreamData<T>,
    dot_sum: Option<T>,
    scalar: T, init: (T, T, T)) {
    let (mut gold_a, mut gold_b, mut gold_c) = init;
    for _ in 0..numtimes {
        match benchmark {
            Benchmark::All => {
                gold_c = gold_a;
                gold_b = scalar * gold_c;
                gold_c = gold_a + gold_b;
                gold_a = gold_b + scalar * gold_c;
            }
            Benchmark::Triad => {
                gold_a = gold_b + scalar * gold_c;
            }
            Benchmark::NStream => {
                gold_a += gold_b + scalar * gold_c;
            }
        };
    }
    let tolerance = T::epsilon().into() * 100.0f64;
    let validate_xs = |name: &str, xs: &Vec<T>, from: T| {
        let error = (xs.iter().map(|x| abs(*x - from)).sum::<T>()).into() / xs.len() as f64;
        if error > tolerance {
            eprintln!("Validation failed on {}[]. Average error {} ", name, error)
        }
    };
    validate_xs("a", &vec.a, gold_a);
    validate_xs("b", &vec.b, gold_b);
    validate_xs("c", &vec.c, gold_c);

    if let Some(sum) = dot_sum {
        let gold_sum = (gold_a * gold_b).into() * vec.size as f64;
        let error = abs((sum.into() - gold_sum) / gold_sum);
        if error > 1.0e-8 {
            eprintln!("Validation failed on sum. Error {} \nSum was {} but should be {}", error, sum, gold_sum);
        }
    }
}

fn run_cpu<T: ArrayType + Sync + Send + Sum + Into<f64> + Display>(option: Options, scalar: T, init: (T, T, T)) {
    let benchmark = match (option.nstream_only, option.triad_only) {
        (true, false) => Benchmark::NStream,
        (false, true) => Benchmark::Triad,
        (false, false) => Benchmark::All,
        (true, true) => panic!("Both triad and nstream are enabled, pick one or omit both to run all benchmarks"),
    };

    let array_bytes = option.arraysize * size_of::<T>();
    let total_bytes = array_bytes * 3;
    let (mega_scale, mega_suffix, giga_scale, giga_suffix) =
        if !option.mibibytes { (1.0e-6, "MB", 1.0e-9, "GB") } else { (2f64.powi(-20), "MiB", 2f64.powi(-30), "GiB") };

    if !option.csv {
        println!("Running {} {} times", match benchmark {
            Benchmark::All => "kernels",
            Benchmark::Triad => "triad",
            Benchmark::NStream => "nstream",
        }, option.numtimes);

        if benchmark == Benchmark::Triad {
            println!("Number of elements: {}", option.arraysize);
        }

        println!("Precision: {}", if option.float { "float" } else { "double" });
        println!("Array size: {:.1} {}(={:.1} {})",
                 mega_scale * array_bytes as f64, mega_suffix, giga_scale * array_bytes as f64, giga_suffix);
        println!("Total size: {:.1} {}(={:.1} {})",
                 mega_scale * total_bytes as f64, mega_suffix, giga_scale * total_bytes as f64, giga_suffix);
    }


    let mut vec: StreamData<T> = StreamData::<T>::new(option.arraysize, scalar);
    let stream = &mut vec as &mut dyn RustStream<T, RayonPar>;
    stream.init_arrays(init);

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
                        (if option.mibibytes { "max_mibytes_per_sec" } else { "max_mbytes_per_sec" }, mbps.to_string()),
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
            (_, _) => panic!("No min/max element for {}(size={})", name, t_size)
        }
    };

    let tabulate_all = |xs: Vec<Vec<(&str, String)>>| {
        match xs.as_slice() {
            [head, ..  ] => {
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
                    println!("{}", table);
                }
            }
            _ => panic!("Empty tabulation")
        };
    };

    match benchmark {
        Benchmark::All => {
            let (results, sum) = stream.run_all(option.numtimes);
            validate(benchmark, option.numtimes, &vec, Some(sum), scalar, init);
            tabulate_all(vec![
                tabulate(&results.copy, "Copy", 2 * array_bytes),
                tabulate(&results.mul, "Mul", 2 * array_bytes),
                tabulate(&results.add, "Add", 3 * array_bytes),
                tabulate(&results.triad, "Triad", 3 * array_bytes),
                tabulate(&results.dot, "Dot", 2 * array_bytes),
            ])
        }
        Benchmark::NStream => {
            let results = stream.run_nstream(option.numtimes);
            validate(benchmark, option.numtimes, &vec, None, scalar, init);
            tabulate_all(vec![
                tabulate(&results, "Nstream", 4 * array_bytes)
            ]);
        }
        Benchmark::Triad => {
            let results = stream.run_triad(option.numtimes);
            let total_bytes = 3 * array_bytes * option.numtimes;
            let bandwidth = mega_scale * (total_bytes as f64 / results.as_secs_f64());

            println!("Runtime (seconds): {:.5}", results.as_secs_f64());
            println!("Bandwidth ({}/s): {:.3} ", giga_suffix, bandwidth);
        }
    };
}

const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

static START_A: f32 = 0.1;
static START_B: f32 = 0.2;
static START_C: f32 = 0.0;
static START_SCALAR: f32 = 0.4;

fn main() {
    let options: Options = Options::from_args();

    // only CPU via Rayon for now
    let devices = vec![("CPU (Rayon)", |opt: Options| {
        if opt.float {
            run_cpu::<f32>(opt, START_SCALAR, (START_A, START_B, START_C));
        } else {
            run_cpu::<f64>(opt, START_SCALAR.into(), (START_A.into(), START_B.into(), START_C.into()));
        }
    })];

    if options.list {
        devices.iter().enumerate().for_each(|(i, (name, _))| {
            println!("{}: {}", i, name);
        })
    } else {
        match devices.get(options.device) {
            Some((_, run)) => {
                if !&options.csv {
                    println!("BabelStream\n\
                              Version: {}\n\
                              Implementation: Rust+Rayon", VERSION.unwrap_or("unknown"))
                }
                run(options);
            }
            None => eprintln!("Device index({}) not available", options.device)
        }
    }
}
