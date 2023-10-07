package javastream.jdk;

import static javastream.FractionalMaths.from;
import static javastream.FractionalMaths.plus;
import static javastream.FractionalMaths.times;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;
import javastream.FractionalMaths;
import javastream.JavaStream;
import javastream.Main.Config;

/**
 * We use
 *
 * <pre>Arrays.parallelSetAll</pre>
 *
 * <p>here as it internally calls
 *
 * <pre>IntStream.range(0, array.length).parallel().forEach(...)</pre>
 */
final class GenericStream<T extends Number> extends JavaStream<T> {

  private final T[] a, b, c;

  @SuppressWarnings("unchecked")
  GenericStream(Config<T> config) {
    super(config);
    this.a = (T[]) Array.newInstance(config.evidence, config.options.arraysize);
    this.b = (T[]) Array.newInstance(config.evidence, config.options.arraysize);
    this.c = (T[]) Array.newInstance(config.evidence, config.options.arraysize);
  }

  @Override
  public List<String> listDevices() {
    return Collections.singletonList("JVM");
  }

  @Override
  public void initArrays() {
    Arrays.parallelSetAll(a, i -> config.initA);
    Arrays.parallelSetAll(b, i -> config.initB);
    Arrays.parallelSetAll(c, i -> config.initC);
  }

  @Override
  public void copy() {
    Arrays.parallelSetAll(c, i -> a[i]);
  }

  @Override
  public void mul() {
    Arrays.parallelSetAll(b, i -> times(config.scalar, c[i]));
  }

  @Override
  public void add() {
    Arrays.parallelSetAll(c, i -> plus(a[i], b[i]));
  }

  @Override
  public void triad() {
    Arrays.parallelSetAll(a, i -> plus(b[i], times(config.scalar, c[i])));
  }

  @Override
  public void nstream() {
    Arrays.parallelSetAll(a, i -> plus(a[i], plus(b[i], times(config.scalar, c[i]))));
  }

  @Override
  public T dot() {
    return IntStream.range(0, config.options.arraysize)
        .parallel()
        .mapToObj(i -> times(a[i], b[i]))
        .reduce(from(config.evidence, 0), FractionalMaths::plus);
  }

  @Override
  public Data<T> readArrays() {
    return new Data<>(a, b, c);
  }
}
