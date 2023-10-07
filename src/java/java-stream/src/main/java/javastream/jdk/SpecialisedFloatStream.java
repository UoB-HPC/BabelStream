package javastream.jdk;

import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;
import javastream.JavaStream;
import javastream.Main.Config;

final class SpecialisedFloatStream extends JavaStream<Float> {

  private final float[] a, b, c;

  SpecialisedFloatStream(Config<Float> config) {
    super(config);
    this.a = new float[config.options.arraysize];
    this.b = new float[config.options.arraysize];
    this.c = new float[config.options.arraysize];
  }

  @Override
  public List<String> listDevices() {
    return Collections.singletonList("JVM");
  }

  @Override
  public void initArrays() {
    IntStream.range(0, config.options.arraysize) //
        .parallel()
        .forEach(
            i -> {
              a[i] = config.initA;
              b[i] = config.initB;
              c[i] = config.initC;
            });
  }

  @Override
  public void copy() {
    IntStream.range(0, config.options.arraysize) //
        .parallel()
        .forEach(i -> c[i] = a[i]);
  }

  @Override
  public void mul() {
    IntStream.range(0, config.options.arraysize) //
        .parallel()
        .forEach(i -> b[i] = config.scalar * c[i]);
  }

  @Override
  public void add() {
    IntStream.range(0, config.options.arraysize) //
        .parallel()
        .forEach(i -> c[i] = a[i] + b[i]);
  }

  @Override
  public void triad() {
    IntStream.range(0, config.options.arraysize) //
        .parallel()
        .forEach(i -> a[i] = b[i] + config.scalar * c[i]);
  }

  @Override
  public void nstream() {
    IntStream.range(0, config.options.arraysize) //
        .parallel()
        .forEach(i -> a[i] += b[i] + config.scalar * c[i]);
  }

  @Override
  public Float dot() {
    return IntStream.range(0, config.options.arraysize) //
        .parallel()
        .mapToObj(i -> a[i] * b[i]) // XXX there isn't a specialised Stream for floats
        .reduce(0f, Float::sum);
  }

  @Override
  public Data<Float> readArrays() {
    return new Data<>(boxed(a), boxed(b), boxed(c));
  }
}
