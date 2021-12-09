package javastream.aparapi;

import static javastream.JavaStream.boxed;

import java.util.Arrays;
import javastream.JavaStream.Data;
import javastream.Main.Config;

final class SpecialisedFloatKernel extends GenericAparapiStreamKernel<Float> {
  private final float scalar;
  final float[] a, b, c;
  private final float[] partialSum;
  @Local private final float[] workGroupSum;

  SpecialisedFloatKernel(Config<Float> config, int numGroups, int workGroupSize) {
    super(config, numGroups, workGroupSize);
    this.scalar = config.scalar;
    this.a = new float[this.arraysize];
    this.b = new float[this.arraysize];
    this.c = new float[this.arraysize];

    this.partialSum = new float[numGroups];
    this.workGroupSum = new float[workGroupSize];
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
      workGroupSum[localId] = 0.f;
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
  public Float dot() {
    partialDot().get(partialSum);
    float sum = 0;
    for (float v : partialSum) sum += v;
    return sum;
  }

  @Override
  public Data<Float> syncAndDispose() {
    get(a).get(b).get(c).dispose();
    return new Data<>(boxed(a), boxed(b), boxed(c));
  }
}
