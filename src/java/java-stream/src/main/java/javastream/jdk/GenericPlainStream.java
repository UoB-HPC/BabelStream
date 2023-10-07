package javastream.jdk;

import static javastream.FractionalMaths.from;
import static javastream.FractionalMaths.plus;
import static javastream.FractionalMaths.times;

import java.lang.reflect.Array;
import java.util.Collections;
import java.util.List;
import javastream.JavaStream;
import javastream.Main.Config;

final class GenericPlainStream<T extends Number> extends JavaStream<T> {

  private final T[] a;
  private final T[] b;
  private final T[] c;

  @SuppressWarnings("unchecked")
  GenericPlainStream(Config<T> config) {
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
      b[i] = times(config.scalar, c[i]);
    }
  }

  @Override
  public void add() {

    for (int i = 0; i < config.options.arraysize; i++) {
      c[i] = plus(a[i], b[i]);
    }
  }

  @Override
  public void triad() {

    for (int i = 0; i < config.options.arraysize; i++) {
      a[i] = plus(b[i], times(config.scalar, c[i]));
    }
  }

  @Override
  public void nstream() {
    for (int i = 0; i < config.options.arraysize; i++) {
      a[i] = plus(a[i], plus(b[i], times(config.scalar, c[i])));
    }
  }

  @Override
  public T dot() {
    T acc = from(config.evidence, 0);
    for (int i = 0; i < config.options.arraysize; i++) {
      acc = plus(acc, times(a[i], b[i]));
    }
    return acc;
  }

  @Override
  public Data<T> readArrays() {
    return new Data<>(a, b, c);
  }
}
