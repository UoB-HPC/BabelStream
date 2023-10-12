package javastream;

import java.time.Duration;
import java.util.AbstractMap;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import javastream.Main.Config;

public abstract class JavaStream<T> {

  public static final class Data<T> {
    final T[] a, b, c;

    public Data(T[] a, T[] b, T[] c) {
      this.a = Objects.requireNonNull(a);
      this.b = Objects.requireNonNull(b);
      this.c = Objects.requireNonNull(c);
    }
  }

  static final class Timings<T> {
    final List<T> copy = new ArrayList<>();
    final List<T> mul = new ArrayList<>();
    final List<T> add = new ArrayList<>();
    final List<T> triad = new ArrayList<>();
    final List<T> dot = new ArrayList<>();
  }

  protected final Config<T> config;

  protected JavaStream(Config<T> config) {
    this.config = config;
  }

  protected abstract List<String> listDevices();

  protected abstract void initArrays();

  protected abstract void copy();

  protected abstract void mul();

  protected abstract void add();

  protected abstract void triad();

  protected abstract void nstream();

  protected abstract T dot();

  protected abstract Data<T> readArrays();

  public static class EnumeratedStream<T> extends JavaStream<T> {

    protected final JavaStream<T> actual;
    private final Entry<String, Function<Config<T>, JavaStream<T>>>[] options;

    @SafeVarargs
    @SuppressWarnings("varargs")
    public EnumeratedStream(
        Config<T> config, Entry<String, Function<Config<T>, JavaStream<T>>>... options) {
      super(config);
      this.actual = options[config.options.device].getValue().apply(config);
      this.options = options;
    }

    @Override
    protected List<String> listDevices() {
      return Arrays.stream(options).map(Entry::getKey).collect(Collectors.toList());
    }

    @Override
    public void initArrays() {
      actual.initArrays();
    }

    @Override
    public void copy() {
      actual.copy();
    }

    @Override
    public void mul() {
      actual.mul();
    }

    @Override
    public void add() {
      actual.add();
    }

    @Override
    public void triad() {
      actual.triad();
    }

    @Override
    public void nstream() {
      actual.nstream();
    }

    @Override
    public T dot() {
      return actual.dot();
    }

    @Override
    public Data<T> readArrays() {
      return actual.readArrays();
    }
  }

  public static Double[] boxed(double[] xs) {
    return Arrays.stream(xs).boxed().toArray(Double[]::new);
  }

  public static Float[] boxed(float[] xs) {
    return IntStream.range(0, xs.length).mapToObj(i -> xs[i]).toArray(Float[]::new);
  }

  private static <T> AbstractMap.SimpleImmutableEntry<Duration, T> timed(Supplier<T> f) {
    long start = System.nanoTime();
    T r = f.get();
    long end = System.nanoTime();
    return new AbstractMap.SimpleImmutableEntry<>(Duration.ofNanos(end - start), r);
  }

  private static Duration timed(Runnable f) {
    long start = System.nanoTime();
    f.run();
    long end = System.nanoTime();
    return Duration.ofNanos(end - start);
  }

  final Duration runInitArrays() {
    return timed(this::initArrays);
  }

  final SimpleImmutableEntry<Duration, Data<T>> runReadArrays() {
    return timed(this::readArrays);
  }

  final SimpleImmutableEntry<Timings<Duration>, T> runAll(int times) {
    Timings<Duration> timings = new Timings<>();
    T lastSum = null;
    for (int i = 0; i < times; i++) {
      timings.copy.add(timed(this::copy));
      timings.mul.add(timed(this::mul));
      timings.add.add(timed(this::add));
      timings.triad.add(timed(this::triad));
      SimpleImmutableEntry<Duration, T> dot = timed(this::dot);
      timings.dot.add(dot.getKey());
      lastSum = dot.getValue();
    }
    return new SimpleImmutableEntry<>(timings, lastSum);
  }

  final Duration runTriad(int times) {
    return timed(
        () -> {
          for (int i = 0; i < times; i++) {
            triad();
          }
        });
  }

  final List<Duration> runNStream(int times) {
    return IntStream.range(0, times)
        .mapToObj(i -> timed(this::nstream))
        .collect(Collectors.toList());
  }
}
