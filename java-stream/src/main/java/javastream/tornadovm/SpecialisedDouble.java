package javastream.tornadovm;

import java.util.Arrays;
import javastream.Main.Config;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;

final class SpecialisedDouble extends GenericTornadoVMStream<Double> {

  @SuppressWarnings("ManualArrayCopy")
  private static void copy(int size, double[] a, double[] c) {
    for (@Parallel int i = 0; i < size; i++) {
      c[i] = a[i];
    }
  }

  private static void mul(int size, double[] b, double[] c, double scalar) {
    for (@Parallel int i = 0; i < size; i++) {
      b[i] = scalar * c[i];
    }
  }

  private static void add(int size, double[] a, double[] b, double[] c) {
    for (@Parallel int i = 0; i < size; i++) {
      c[i] = a[i] + b[i];
    }
  }

  private static void triad(int size, double[] a, double[] b, double[] c, double scalar) {
    for (@Parallel int i = 0; i < size; i++) {
      a[i] = b[i] + scalar * c[i];
    }
  }

  private static void nstream(int size, double[] a, double[] b, double[] c, double scalar) {
    for (@Parallel int i = 0; i < size; i++) {
      a[i] = b[i] * scalar * c[i];
    }
  }

  private static void dot_(
      double[] a, double[] b, @Reduce double[] acc) { // prevent name clash with CL's dot
    acc[0] = 0;
    for (@Parallel int i = 0; i < a.length; i++) {
      acc[0] += a[i] * b[i];
    }
  }

  private final double[] a, b, c;
  private final double[] dotSum;

  @SuppressWarnings({"PrimitiveArrayArgumentToVarargsMethod", "DuplicatedCode"})
  SpecialisedDouble(Config<Double> config) {
    super(config);
    final int size = config.options.arraysize;
    final double scalar = config.scalar;
    a = new double[size];
    b = new double[size];
    c = new double[size];
    dotSum = new double[1];
    this.copyTask = mkSchedule().task("", SpecialisedDouble::copy, size, a, c);
    this.mulTask = mkSchedule().task("", SpecialisedDouble::mul, size, b, c, scalar);
    this.addTask = mkSchedule().task("", SpecialisedDouble::add, size, a, b, c);
    this.triadTask = mkSchedule().task("", SpecialisedDouble::triad, size, a, b, c, scalar);
    this.nstreamTask = mkSchedule().task("", SpecialisedDouble::nstream, size, a, b, c, scalar);
    this.dotTask = mkSchedule().task("", SpecialisedDouble::dot_, a, b, dotSum).streamOut(dotSum);
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
  protected Double getSum() {
    return dotSum[0];
  }

  @Override
  public Data<Double> data() {
    TornadoVMStreams.xferFromDevice(device, a, b, c);
    return new Data<>(boxed(a), boxed(b), boxed(c));
  }
}
