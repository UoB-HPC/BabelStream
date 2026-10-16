rust-stream
===========

This is an implementation of BabelStream in Rust.

Currently, we support five CPU threading APIs as devices:

* Plain - basic single-threaded `for` version, see [plain_stream.rs](src/plain_stream.rs)
* [Rayon](https://github.com/rayon-rs/rayon) - Parallel with high level API,
  see [rayon_stream.rs](src/rayon_stream.rs)
* [Crossbeam](https://github.com/crossbeam-rs/crossbeam) - Parallel with partitions per thread,
  see [crossbeam_stream.rs](src/crossbeam_stream.rs)
* Arc - Parallel with `Vec` per thread (static partitions) wrapped in `Mutex` contained in `Arc`s,
  see [crossbeam_stream.rs](src/arc_stream.rs)
* Unsafe - Parallel with unsafe pointer per thread (static partitions) to `Vec`,
  see [crossbeam_stream.rs](src/unsafe_stream.rs)

In addition, this implementation also supports the following extra flags:
****
```
--init    Initialise each benchmark array at allocation time on the main thread
--malloc  Use libc calloc instead of the Rust allocator for benchmark array allocation
--pin     Pin threads to distinct cores, this has NO effect in Rayon devices
```

Max thread count is controlled by the environment variable `BABELSTREAM_NUM_THREADS` which is compatible for all devices (avoid setting `RAYON_NUM_THREADS`, the implementation will issue a warning if this happened).   

### Build & Run

Prerequisites:

* [Rust toolchain](https://www.rust-lang.org/tools/install), stable 1.93 or later

With `cargo` on path, compile and run the benchmark with:

```shell
> cd rust-stream/
> cargo build --release
> ./target/release/rust-stream --help
Usage: rust-stream [OPTIONS]

Options:
      --list                   List available devices
      --device <DEVICE>        Select device at <device> [default: 0]
  -n, --numtimes <NUMTIMES>    Run the test <numtimes> times (NUM >= 2) [default: 100]
  -s, --arraysize <ARRAYSIZE>  Use <arraysize> elements in the array [default: 33554432]
      --float                  Use floats (rather than doubles)
      --triad-only             Only run triad
      --nstream-only           Only run nstream
      --csv                    Output as csv table
      --mibibytes              Use MiB=2^20 for bandwidth calculation (default MB=10^6)
      --malloc                 Use libc calloc instead of the Rust allocator for benchmark array allocation
      --init                   Initialise each benchmark array at allocation time on the main thread
      --pin                    Pin threads to distinct cores, this has NO effect in Rayon devices
  -h, --help                   Print help
  -V, --version                Print version
```
