package javastream.tornadovm;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import javastream.JavaStream;
import javastream.Main.Config;
import uk.ac.manchester.tornado.api.TornadoRuntimeCI;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.mm.TornadoGlobalObjectState;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntime;

public final class TornadoVMStreams {

  private TornadoVMStreams() {}

  static void xferToDevice(TornadoDevice device, Object... xs) {
    for (Object x : xs) {
      TornadoGlobalObjectState state = TornadoRuntime.getTornadoRuntime().resolveObject(x);
      List<Integer> writeEvent = device.ensurePresent(x, state.getDeviceState(device), null, 0, 0);
      if (writeEvent != null) writeEvent.forEach(e -> device.resolveEvent(e).waitOn());
    }
  }

  static void xferFromDevice(TornadoDevice device, Object... xs) {
    for (Object x : xs) {
      TornadoGlobalObjectState state = TornadoRuntime.getTornadoRuntime().resolveObject(x);
      device.resolveEvent(device.streamOut(x, 0, state.getDeviceState(device), null)).waitOn();
    }
  }

  static List<TornadoDevice> enumerateDevices(TornadoRuntimeCI runtime) {
    return IntStream.range(0, runtime.getNumDrivers())
        .mapToObj(runtime::getDriver)
        .flatMap(d -> IntStream.range(0, d.getDeviceCount()).mapToObj(d::getDevice))
        .collect(Collectors.toList());
  }

  public static final Function<Config<Float>, JavaStream<Float>> FLOAT = SpecialisedFloat::new;
  public static final Function<Config<Double>, JavaStream<Double>> DOUBLE = SpecialisedDouble::new;
}
