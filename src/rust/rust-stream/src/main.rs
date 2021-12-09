fn main() {
  if !rust_stream::run(&std::env::args().collect::<Vec<_>>()) {
    std::process::exit(1);
  }
}
