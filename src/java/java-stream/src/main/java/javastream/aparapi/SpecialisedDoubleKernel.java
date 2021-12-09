package javastream.aparapi;

import java.util.Arrays;
import javastream.JavaStream;
import javastream.JavaStream.Data;
import javastream.Main.Config;

final class SpecialisedDoubleKernel extends GenericAparapiStreamKernel<Double> {
  private final double scalar;
  final double[] a, b, c;
  private final double[] partialSum;
  @Local private final double[] workGroupSum;

  SpecialisedDoubleKernel(Config<Double> config, int numGroups, int workGroupSize) {
    super(config, numGroups, workGroupSize);
    this.scalar = config.scalar;
    this.a = new double[this.arraysize];
    this.b = new double[this.arraysize];
    this.c = new double[this.arraysize];

    this.partialSum = new double[numGroups];
    this.workGroupSum = new double[workGroupSize];
  }

  @SuppressWarnings("DuplicatedCode")
  @Override
  public void run() {
    int i = getGlobalId();
    if (function == FN_COPY) {
      c[i] = a[i];
    } else if (function == FN_MUL) {
      b[i] = scalar * c[i];
    } else if (function == FN_ADD) {
      c[i] = a[i] + b[i];
    } else if (function == FN_TRIAD) {
      a[i] = b[i] + scalar * c[i];
    } else if (function == FN_NSTREAM) {
      a[i] += b[i] + scalar * c[i];
    } else if (function == FN_DOT) {
      int localId = getLocalId(0);
      workGroupSum[localId] = 0.0;
      for (; i < arraysize; i += getGlobalSize(0)) workGroupSum[localId] += a[i] * b[i];
      for (int offset = getLocalSize(0) / 2; offset > 0; offset /= 2) {
        localBarrier();
        if (localId < offset) {
          workGroupSum[localId] += workGroupSum[localId + offset];
        }
      }
      if (localId == 0) partialSum[getGroupId(0)] = workGroupSum[localId];
    }
  }

  @Override
  public void init() {
    Arrays.fill(a, config.initA);
    Arrays.fill(b, config.initB);
    Arrays.fill(c, config.initC);
    put(a).put(b).put(c);
  }

  @Override
  public Double dot() {
    partialDot().get(partialSum);
    double sum = 0;
    for (double v : partialSum) sum += v;
    return sum;
  }

  @Override
  public Data<Double> syncAndDispose() {
    get(a).get(b).get(c).dispose();
    return new Data<>(JavaStream.boxed(a), JavaStream.boxed(b), JavaStream.boxed(c));
  }
}
