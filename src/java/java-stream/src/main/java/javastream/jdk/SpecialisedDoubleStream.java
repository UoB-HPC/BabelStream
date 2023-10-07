package javastream.jdk;

import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;
import javastream.JavaStream;
import javastream.Main.Config;

final class SpecialisedDoubleStream extends JavaStream<Double> {

  private final double[] a, b, c;

  SpecialisedDoubleStream(Config<Double> config) {
    super(config);
    this.a = new double[config.options.arraysize];
    this.b = new double[config.options.arraysize];
    this.c = new double[config.options.arraysize];
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
  public Double dot() {
    return IntStream.range(0, config.options.arraysize)
        .parallel()
        .mapToDouble(i -> a[i] * b[i])
        .reduce(0f, Double::sum);
  }

  @Override
  public Data<Double> readArrays() {
    return new Data<>(boxed(a), boxed(b), boxed(c));
  }
}
