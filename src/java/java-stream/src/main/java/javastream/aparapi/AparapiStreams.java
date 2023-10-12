package javastream.aparapi;

import com.aparapi.device.Device;
import com.aparapi.device.Device.TYPE;
import com.aparapi.device.JavaDevice;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javastream.JavaStream;
import javastream.Main.Config;

public final class AparapiStreams {

  private AparapiStreams() {}

  public static final Function<Config<Double>, JavaStream<Double>> DOUBLE =
      config -> new Generic<>(config, SpecialisedDoubleKernel::new);

  public static final Function<Config<Float>, JavaStream<Float>> FLOAT =
      config -> new Generic<>(config, SpecialisedFloatKernel::new);

  private static List<Device> enumerateDevices() {

    // JavaDevice.SEQUENTIAL doesn't work when arraysize > 1, so we omit it entirely
    Stream<JavaDevice> cpuDevices = Stream.of(JavaDevice.ALTERNATIVE_ALGORITHM);

    Stream<OpenCLDevice> clDevices =
        Stream.of(TYPE.values()).map(OpenCLDevice::listDevices).flatMap(Collection::stream);

    return Stream.concat(clDevices, cpuDevices).collect(Collectors.toList());
  }

  private static String deviceName(Device device) {
    return device.toString();
  }

  private static final class Generic<T extends Number> extends JavaStream<T> {

    private final GenericAparapiStreamKernel<T> kernels;

    Generic(Config<T> config, GenericAparapiStreamKernel.Factory<T> factory) {
      super(config);
      Device device = enumerateDevices().get(config.options.device);

      final int numGroups;
      final int workGroupSize;
      if (device instanceof JavaDevice) {
        numGroups = Runtime.getRuntime().availableProcessors();
        workGroupSize =
            config.typeSize * 2; // closest thing to CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE

      } else if (device instanceof OpenCLDevice) {
        numGroups = ((OpenCLDevice) device).getMaxComputeUnits();
        workGroupSize = device.getMaxWorkGroupSize();
      } else {
        throw new AssertionError("Unknown device type " + device.getClass());
      }

      if (config.options.isVerboseBenchmark()) {
        System.out.println("Using Aparapi OpenCL device: " + device);
        System.out.println(" - numGroups     : " + numGroups);
        System.out.println(" - workGroupSize : " + workGroupSize);
        String showCL = System.getProperty("com.aparapi.enableShowGeneratedOpenCL");
        if (showCL == null || !showCL.equals("true")) {
          System.out.println(
              "(Add `-Dcom.aparapi.enableShowGeneratedOpenCL=true` to show generated OpenCL source)");
        }
      }

      LinkedHashSet<Device> candidate = new LinkedHashSet<>();
      candidate.add(device);

      kernels = factory.create(config, numGroups, workGroupSize);
      KernelManager.instance().setPreferredDevices(kernels, candidate);
    }

    @Override
    public List<String> listDevices() {
      return enumerateDevices().stream()
          .map(AparapiStreams::deviceName)
          .collect(Collectors.toList());
    }

    @Override
    public void initArrays() {
      kernels.init();
    }

    @Override
    public void copy() {
      kernels.copy();
    }

    @Override
    public void mul() {
      kernels.mul();
    }

    @Override
    public void add() {
      kernels.add();
    }

    @Override
    public void triad() {
      kernels.triad();
    }

    @Override
    public void nstream() {
      kernels.nstream();
    }

    @Override
    public T dot() {
      return kernels.dot();
    }

    @Override
    public Data<T> readArrays() {
      return kernels.syncAndDispose();
    }
  }
}
