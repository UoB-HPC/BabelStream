package javastream.jdk;

import java.util.Collections;
import java.util.List;
import javastream.JavaStream;
import javastream.Main.Config;

final class SpecialisedPlainDoubleStream extends JavaStream<Double> {

  private final double[] a;
  private final double[] b;
  private final double[] c;

  SpecialisedPlainDoubleStream(Config<Double> config) {
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
    for (int i = 0; i < config.options.arraysize; i++) {
      a[i] = config.initA;
      b[i] = config.initB;
      c[i] = config.initC;
    }
  }

  @SuppressWarnings("ManualArrayCopy")
  @Override
  public void copy() {
    for (int i = 0; i < config.options.arraysize; i++) {
      c[i] = a[i];
    }
  }

  @Override
  public void mul() {
    for (int i = 0; i < config.options.arraysize; i++) {
      b[i] = config.scalar * c[i];
    }
  }

  @Override
  public void add() {
    for (int i = 0; i < config.options.arraysize; i++) {
      c[i] = a[i] + b[i];
    }
  }

  @Override
  public void triad() {
    for (int i = 0; i < config.options.arraysize; i++) {
      a[i] = b[i] + config.scalar * c[i];
    }
  }

  @Override
  public void nstream() {
    for (int i = 0; i < config.options.arraysize; i++) {
      a[i] += b[i] + config.scalar * c[i];
    }
  }

  @Override
  public Double dot() {
    double acc = 0f;
    for (int i = 0; i < config.options.arraysize; i++) {
      acc += a[i] * b[i];
    }
    return acc;
  }

  @Override
  public Data<Double> readArrays() {
    return new Data<>(boxed(a), boxed(b), boxed(c));
  }
}
