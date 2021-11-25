package javastream.tornadovm;

import java.util.List;
import java.util.stream.Collectors;
import javastream.JavaStream;
import javastream.Main.Config;
import uk.ac.manchester.tornado.api.TaskSchedule;
import uk.ac.manchester.tornado.api.TornadoRuntimeCI;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntime;

abstract class GenericTornadoVMStream<T> extends JavaStream<T> {

  protected final TornadoDevice device;

  protected TaskSchedule copyTask;
  protected TaskSchedule mulTask;
  protected TaskSchedule addTask;
  protected TaskSchedule triadTask;
  protected TaskSchedule nstreamTask;
  protected TaskSchedule dotTask;

  GenericTornadoVMStream(Config<T> config) {
    super(config);

    try {
      TornadoRuntimeCI runtime = TornadoRuntime.getTornadoRuntime();
      List<TornadoDevice> devices = TornadoVMStreams.enumerateDevices(runtime);
      device = devices.get(config.options.device);

      if (config.options.isVerboseBenchmark()) {
        System.out.println("Using TornadoVM device:");
        System.out.println(" - Name     : " + device.getDescription());
        System.out.println(" - Id       : " + device.getDeviceName());
        System.out.println(" - Platform : " + device.getPlatformName());
        System.out.println(" - Backend  : " + device.getTornadoVMBackend().name());
      }
    } catch (Throwable e) {
      throw new RuntimeException(
          "Unable to initialise TornadoVM, make sure you are running the binary with the `tornado -jar ...` wrapper and not `java -jar ...`",
          e);
    }
  }

  protected static TaskSchedule mkSchedule() {
    return new TaskSchedule("");
  }

  @Override
  public List<String> listDevices() {
    return TornadoVMStreams.enumerateDevices(TornadoRuntime.getTornadoRuntime()).stream()
        .map(d -> d.getDescription() + "(" + d.getDeviceName() + ")")
        .collect(Collectors.toList());
  }

  @Override
  public void initArrays() {
    this.copyTask.warmup();
    this.mulTask.warmup();
    this.addTask.warmup();
    this.triadTask.warmup();
    this.nstreamTask.warmup();
    this.dotTask.warmup();
  }

  @Override
  public void copy() {
    this.copyTask.execute();
  }

  @Override
  public void mul() {
    this.mulTask.execute();
  }

  @Override
  public void add() {
    this.addTask.execute();
  }

  @Override
  public void triad() {
    this.triadTask.execute();
  }

  @Override
  public void nstream() {
    this.nstreamTask.execute();
  }

  protected abstract T getSum();

  @Override
  public T dot() {
    this.dotTask.execute();
    return getSum();
  }
}
