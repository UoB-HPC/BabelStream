use rstest::rstest;

#[rstest]
fn test_main(
  #[values(0, 1, 2, 3, 4)] device: usize, //
  #[values("", "--pin")] pin: &str,       //
  #[values("", "--malloc")] malloc: &str, //
  #[values("", "--init")] init: &str,     //
  #[values("", "--triad-only", "--nstream-only")] option: &str, //
) {
  let line = format!(
    "rust-stream --arraysize 2048 --device {} {} {} {} {}",
    device, pin, malloc, init, option
  );
  let args = line.split_whitespace().map(|s| s.to_string()).collect::<Vec<_>>();
  assert!(rust_stream::run(&args));
}
