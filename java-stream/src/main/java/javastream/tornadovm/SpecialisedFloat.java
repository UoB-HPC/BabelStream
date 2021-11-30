package javastream.tornadovm;

import java.util.Arrays;
import javastream.Main.Config;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;

final class SpecialisedFloat extends GenericTornadoVMStream<Float> {

  @SuppressWarnings("ManualArrayCopy")
  private static void copy(int size, float[] a, float[] c) {
    for (@Parallel int i = 0; i < size; i++) {
      c[i] = a[i];
    }
  }

  private static void mul(int size, float[] b, float[] c, float scalar) {
    for (@Parallel int i = 0; i < size; i++) {
      b[i] = scalar * c[i];
    }
  }

  private static void add(int size, float[] a, float[] b, float[] c) {
    for (@Parallel int i = 0; i < size; i++) {
      c[i] = a[i] + b[i];
    }
  }

  private static void triad(int size, float[] a, float[] b, float[] c, float scalar) {
    for (@Parallel int i = 0; i < size; i++) {
      a[i] = b[i] + scalar * c[i];
    }
  }

  private static void nstream(int size, float[] a, float[] b, float[] c, float scalar) {
    for (@Parallel int i = 0; i < size; i++) {
      a[i] = b[i] * scalar * c[i];
    }
  }

  private static void dot_(
      float[] a, float[] b, @Reduce float[] acc) { // prevent name clash with CL's dot
    acc[0] = 0;
    for (@Parallel int i = 0; i < a.length; i++) {
      acc[0] += a[i] * b[i];
    }
  }

  private final float[] a, b, c;
  private final float[] dotSum;

  @SuppressWarnings({"PrimitiveArrayArgumentToVarargsMethod", "DuplicatedCode"})
  SpecialisedFloat(Config<Float> config) {
    super(config);
    final int size = config.options.arraysize;
    final float scalar = config.scalar;
    a = new float[size];
    b = new float[size];
    c = new float[size];
    dotSum = new float[1];
    this.copyTask = mkSchedule().task("", SpecialisedFloat::copy, size, a, c);
    this.mulTask = mkSchedule().task("", SpecialisedFloat::mul, size, b, c, scalar);
    this.addTask = mkSchedule().task("", SpecialisedFloat::add, size, a, b, c);
    this.triadTask = mkSchedule().task("", SpecialisedFloat::triad, size, a, b, c, scalar);
    this.nstreamTask = mkSchedule().task("", SpecialisedFloat::nstream, size, a, b, c, scalar);
    this.dotTask = mkSchedule().task("", SpecialisedFloat::dot_, a, b, dotSum).streamOut(dotSum);
  }

  @Override
  public void initArrays() {
    super.initArrays();
    Arrays.fill(a, config.initA);
    Arrays.fill(b, config.initB);
    Arrays.fill(c, config.initC);
    TornadoVMStreams.xferToDevice(device, a, b, c);
  }

  @Override
  protected Float getSum() {
    return dotSum[0];
  }

  @Override
  public Data<Float> data() {
    TornadoVMStreams.xferFromDevice(device, a, b, c);
    return new Data<>(boxed(a), boxed(b), boxed(c));
  }
}
