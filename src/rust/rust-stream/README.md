rust-stream
===========

This is an implementation of BabelStream in Rust.

Currently, we support three CPU threading API as devices:

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
--malloc  Use libc malloc instead of the Rust's allocator for benchmark array allocation
--pin     Pin threads to distinct cores, this has NO effect in Rayon devices
```

Max thread count is controlled by the environment variable `BABELSTREAM_NUM_THREADS` which is compatible for all devices (avoid setting `RAYON_NUM_THREADS`, the implementation will issue a warning if this happened).   

There is an ongoing investigation on potential performance issues under NUMA situations. As part of
the experiment, this implementation made use of the
provisional [Allocator traits](https://github.com/rust-lang/rust/issues/32838) which requires rust
unstable. We hope a NUMA aware allocator will be available once the allocator API reaches rust
stable.

### Build & Run

Prerequisites:

* [Rust toolchain](https://www.rust-lang.org/tools/install)

Once the toolchain is installed, enable the nightly channel:

```shell
> rustup install nightly
> rustup default nightly # optional, this sets `+nightly` automatically for cargo calls later
```

With `cargo` on path, compile and run the benchmark with:

```shell
> cd rust-stream/
> cargo +nightly build --release # or simply `cargo build --release` if nightly channel is the default 
> ./target/release/rust-stream --help
rust-stream 3.4.0

USAGE:
    rust-stream [FLAGS] [OPTIONS]

FLAGS:
        --csv             Output as csv table
        --float           Use floats (rather than doubles)
    -h, --help            Prints help information
        --init            Initialise each benchmark array at allocation time on the main thread
        --list            List available devices
        --malloc          Use libc malloc instead of the Rust's allocator for benchmark array allocation
        --mibibytes       Use MiB=2^20 for bandwidth calculation (default MB=10^6)
        --nstream-only    Only run nstream
        --pin             Pin threads to distinct cores, this has NO effect in Rayon devices
        --triad-only      Only run triad
    -V, --version         Prints version information

OPTIONS:
    -s, --arraysize <arraysize>    Use <arraysize> elements in the array [default: 33554432]
        --device <device>          Select device at <device> [default: 0]
    -n, --numtimes <numtimes>      Run the test <numtimes> times (NUM >= 2) [default: 100]
```


 