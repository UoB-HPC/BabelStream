package javastream.tornadovm;

import java.util.Arrays;
import javastream.Main.Config;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

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

  @SuppressWarnings({"DuplicatedCode"})
  SpecialisedFloat(Config<Float> config) {
    super(config);
    final int size = config.options.arraysize;
    final float scalar = config.scalar;
    a = new float[size];
    b = new float[size];
    c = new float[size];
    dotSum = new float[1];
    this.copyTask =
        new TornadoExecutionPlan(
            new TaskGraph("copy")
                .task("copy", SpecialisedFloat::copy, size, a, c)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, c)
                .snapshot());
    this.mulTask =
        new TornadoExecutionPlan(
            new TaskGraph("mul")
                .task("mul", SpecialisedFloat::mul, size, b, c, scalar)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, b, c)
                .snapshot());
    this.addTask =
        new TornadoExecutionPlan(
            new TaskGraph("add")
                .task("add", SpecialisedFloat::add, size, a, b, c)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, c)
                .snapshot());
    this.triadTask =
        new TornadoExecutionPlan(
            new TaskGraph("triad")
                .task("triad", SpecialisedFloat::triad, size, a, b, c, scalar)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, c)
                .snapshot());
    this.nstreamTask =
        new TornadoExecutionPlan(
            new TaskGraph("nstream")
                .task("nstream", SpecialisedFloat::nstream, size, a, b, c, scalar)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, c)
                .snapshot());
    this.dotTask =
        new TornadoExecutionPlan(
            new TaskGraph("dot")
                .task("dot", SpecialisedFloat::dot_, a, b, dotSum)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, new Object[] {dotSum})
                .snapshot());
  }

  @Override
  public void initArrays() {
    super.initArrays();
    Arrays.fill(a, config.initA);
    Arrays.fill(b, config.initB);
    Arrays.fill(c, config.initC);
    TornadoVMStreams.allocAndXferToDevice(device, a, b, c);
  }

  @Override
  protected Float getSum() {
    return dotSum[0];
  }

  @Override
  public Data<Float> readArrays() {
    TornadoVMStreams.xferFromDevice(device, a, b, c);
    return new Data<>(boxed(a), boxed(b), boxed(c));
  }
}
