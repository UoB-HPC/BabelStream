package javastream;

import static javastream.FractionalMaths.divide;
import static javastream.FractionalMaths.from;
import static javastream.FractionalMaths.minus;
import static javastream.FractionalMaths.plus;
import static javastream.FractionalMaths.times;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import java.time.Duration;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import javastream.JavaStream.Data;
import javastream.JavaStream.Timings;
import javastream.aparapi.AparapiStreams;
import javastream.jdk.JdkStreams;
import javastream.jdk.PlainStream;
import javastream.tornadovm.TornadoVMStreams;

public class Main {

  enum Benchmark {
    NSTREAM,
    TRIAD,
    ALL
  }

  public static class Options {

    @Parameter(names = "--list", description = "List available devices for all implementations")
    boolean list = false;

    @Parameter(
        names = "--device",
        description = "Select device at <device>, see --list for options")
    public int device = 0;

    @Parameter(
        names = "--impl",
        description = "Select implementation at <impl>, see --list for options")
    public String impl = "";

    @Parameter(
        names = {"--numtimes", "-n"},
        description = "Run the test <numtimes> times (NUM >= 2)")
    public int numtimes = 100;

    @Parameter(
        names = {"--arraysize", "-s"},
        description = "Use <arraysize> elements in the array")
    public int arraysize = 33554432;

    @Parameter(names = "--float", description = "Use floats (rather than doubles)")
    public boolean useFloat = false;

    @Parameter(names = "--triad-only", description = "Only run triad")
    public boolean triadOnly = false;

    @Parameter(names = "--nstream-only", description = "Only run nstream")
    public boolean nstreamOnly = false;

    @Parameter(names = "--csv", description = "Output as csv table")
    public boolean csv = false;

    @Parameter(
        names = "--mibibytes",
        description = "Use MiB=2^20 for bandwidth calculation (default MB=10^6)")
    public boolean mibibytes = false;

    @Parameter(names = "--dot-tolerance", description = "Tolerance for dot kernel verification")
    public double dotTolerance = 1.0e-8;

    public boolean isVerboseBenchmark() {
      return !list && !csv;
    }
  }

  public static final class Config<T> {
    public final Options options;
    public final Benchmark benchmark;
    public final int typeSize;
    public final Class<T> evidence;
    public final T ulp, scalar, initA, initB, initC;

    public Config(
        Options options,
        Benchmark benchmark,
        int typeSize,
        Class<T> evidence,
        T ulp,
        T scalar,
        T initA,
        T initB,
        T initC) {
      this.options = Objects.requireNonNull(options);
      this.benchmark = Objects.requireNonNull(benchmark);
      this.typeSize = typeSize;
      this.evidence = Objects.requireNonNull(evidence);
      this.ulp = Objects.requireNonNull(ulp);
      this.scalar = Objects.requireNonNull(scalar);
      this.initA = Objects.requireNonNull(initA);
      this.initB = Objects.requireNonNull(initB);
      this.initC = Objects.requireNonNull(initC);
    }
  }

  static final class Implementation {
    final String name;
    final Function<Config<Float>, JavaStream<Float>> makeFloat;
    final Function<Config<Double>, JavaStream<Double>> makeDouble;

    Implementation(
        String name,
        Function<Config<Float>, JavaStream<Float>> makeFloat,
        Function<Config<Double>, JavaStream<Double>> makeDouble) {
      this.name = Objects.requireNonNull(name);
      this.makeFloat = Objects.requireNonNull(makeFloat);
      this.makeDouble = Objects.requireNonNull(makeDouble);
    }
  }

  @SuppressWarnings("unchecked")
  static void showInit(
      int totalBytes, double megaScale, Options opt, Duration init, Duration read) {
    List<Entry<String, Double>> setup =
        Arrays.asList(
            new SimpleImmutableEntry<>("Init", durationToSeconds(init)),
            new SimpleImmutableEntry<>("Read", durationToSeconds(read)));
    if (opt.csv) {
      tabulateCsv(
          true,
          setup.stream()
              .map(
                  x ->
                      Arrays.asList(
                          new SimpleImmutableEntry<>("function", x.getKey()),
                          new SimpleImmutableEntry<>("n_elements", opt.arraysize + ""),
                          new SimpleImmutableEntry<>("sizeof", totalBytes + ""),
                          new SimpleImmutableEntry<>(
                              "max_m" + (opt.mibibytes ? "i" : "") + "bytes_per_sec",
                              ((megaScale * (double) totalBytes / x.getValue())) + ""),
                          new SimpleImmutableEntry<>("runtime", x.getValue() + "")))
              .toArray(List[]::new));
    } else {
      for (Entry<String, Double> e : setup) {
        System.out.printf(
            "%s: %.5f s (%.5f M%sBytes/sec)%n",
            e.getKey(),
            e.getValue(),
            megaScale * (double) totalBytes / e.getValue(),
            opt.mibibytes ? "i" : "");
      }
    }
  }

  static <T extends Number> boolean run(
      String name, Config<T> config, Function<Config<T>, JavaStream<T>> mkStream) {

    Options opt = config.options;

    int arrayBytes = opt.arraysize * config.typeSize;
    int totalBytes = arrayBytes * 3;

    String megaSuffix = opt.mibibytes ? "MiB" : "MB";
    String gigaSuffix = opt.mibibytes ? "GiB" : "GB";

    double megaScale = opt.mibibytes ? Math.pow(2.0, -20) : 1.0e-6;
    double gigaScale = opt.mibibytes ? Math.pow(2.0, -30) : 1.0e-9;

    if (!opt.csv) {

      String vendor = System.getProperty("java.vendor");
      String ver = System.getProperty("java.version");
      String home = System.getProperty("java.home");

      System.out.println("BabelStream");
      System.out.printf("Version: %s%n", VERSION);
      System.out.printf(
          "Implementation: %s (Java %s; %s; JAVA_HOME=%s)%n", name, ver, vendor, home);
      final String benchmarkName;
      switch (config.benchmark) {
        case NSTREAM:
          benchmarkName = "nstream";
          break;
        case TRIAD:
          benchmarkName = "triad";
          break;
        case ALL:
          benchmarkName = "all";
          break;
        default:
          throw new AssertionError("Unexpected value: " + config.benchmark);
      }
      System.out.println("Running " + benchmarkName + " " + opt.numtimes + " times");

      if (config.benchmark == Benchmark.TRIAD) {
        System.out.println("Number of elements: " + opt.arraysize);
      }

      System.out.println("Precision: " + (opt.useFloat ? "float" : "double"));
      System.out.printf(
          "Array size: %.1f %s (=%.1f %s)%n",
          (megaScale * arrayBytes), megaSuffix, (gigaScale * arrayBytes), gigaSuffix);
      System.out.printf(
          "Total size: %.1f %s (=%.1f %s)%n",
          (megaScale * totalBytes), megaSuffix, (gigaScale * totalBytes), gigaSuffix);
    }

    JavaStream<T> stream = mkStream.apply(config);

    Duration init = stream.runInitArrays();
    final boolean ok;
    switch (config.benchmark) {
      case ALL:
        {
          Entry<Timings<Duration>, T> results = stream.runAll(opt.numtimes);
          SimpleImmutableEntry<Duration, Data<T>> read = stream.runReadArrays();
          showInit(totalBytes, megaScale, opt, init, read.getKey());
          ok = checkSolutions(read.getValue(), config, Optional.of(results.getValue()));
          Timings<Duration> timings = results.getKey();
          tabulateCsv(
              opt.csv,
              mkCsvRow(timings.copy, "Copy", 2 * arrayBytes, megaScale, opt),
              mkCsvRow(timings.mul, "Mul", 2 * arrayBytes, megaScale, opt),
              mkCsvRow(timings.add, "Add", 3 * arrayBytes, megaScale, opt),
              mkCsvRow(timings.triad, "Triad", 3 * arrayBytes, megaScale, opt),
              mkCsvRow(timings.dot, "Dot", 2 * arrayBytes, megaScale, opt));
          break;
        }
      case NSTREAM:
        {
          List<Duration> nstreamResults = stream.runNStream(opt.numtimes);
          SimpleImmutableEntry<Duration, Data<T>> read = stream.runReadArrays();
          showInit(totalBytes, megaScale, opt, init, read.getKey());
          ok = checkSolutions(read.getValue(), config, Optional.empty());
          tabulateCsv(opt.csv, mkCsvRow(nstreamResults, "Nstream", 4 * arrayBytes, megaScale, opt));
          break;
        }
      case TRIAD:
        {
          Duration triadResult = stream.runTriad(opt.numtimes);
          SimpleImmutableEntry<Duration, Data<T>> read = stream.runReadArrays();
          showInit(totalBytes, megaScale, opt, init, read.getKey());
          ok = checkSolutions(read.getValue(), config, Optional.empty());
          int triadTotalBytes = 3 * arrayBytes * opt.numtimes;
          double bandwidth = megaScale * (triadTotalBytes / durationToSeconds(triadResult));
          System.out.printf("Runtime (seconds): %.5f", durationToSeconds(triadResult));
          System.out.printf("Bandwidth (%s/s): %.3f ", gigaSuffix, bandwidth);
          break;
        }
      default:
        throw new AssertionError();
    }
    return ok;
  }

  private static <T extends Number> boolean checkWithinTolerance(
      String name, T[] xs, T gold, T tolerance) {
    // it's ok to default to double for error calculation
    double error =
        Arrays.stream(xs)
            .mapToDouble(x -> Math.abs(minus(x, gold).doubleValue()))
            .summaryStatistics()
            .getAverage();
    boolean failed = error > tolerance.doubleValue();
    if (failed) {
      System.err.printf("Validation failed on %s. Average error %s%n", name, error);
    }
    return !failed;
  }

  @SuppressWarnings("OptionalUsedAsFieldOrParameterType")
  static <T extends Number> boolean checkSolutions(
      Data<T> data, Config<T> config, Optional<T> dotSum) {
    T goldA = config.initA;
    T goldB = config.initB;
    T goldC = config.initC;

    for (int i = 0; i < config.options.numtimes; i++) {
      switch (config.benchmark) {
        case ALL:
          goldC = goldA;
          goldB = times(config.scalar, goldC);
          goldC = plus(goldA, goldB);
          goldA = plus(goldB, times(config.scalar, goldC));
          break;
        case TRIAD:
          goldA = plus(goldB, times(config.scalar, goldC));
          break;
        case NSTREAM:
          goldA = plus(goldA, plus(goldB, times(config.scalar, goldC)));
          break;
      }
    }

    T tolerance = times(config.ulp, from(config.evidence, 100));
    boolean aValid = checkWithinTolerance("a", data.a, goldA, tolerance);
    boolean bValid = checkWithinTolerance("b", data.b, goldB, tolerance);
    boolean cValid = checkWithinTolerance("c", data.c, goldC, tolerance);

    final T finalGoldA = goldA;
    final T finalGoldB = goldB;
    boolean sumValid =
        dotSum
            .map(
                actual -> {
                  T goldSum =
                      times(
                          times(finalGoldA, finalGoldB),
                          from(config.evidence, config.options.arraysize));
                  double error = Math.abs(divide(minus(actual, goldSum), goldSum).doubleValue());
                  boolean failed = error > config.options.dotTolerance;
                  if (failed) {
                    System.err.printf(
                        "Validation failed on sum. Error %s \nSum was %s but should be %s%n",
                        error, actual, goldSum);
                  }
                  return !failed;
                })
            .orElse(true);

    return aValid && bValid && cValid && sumValid;
  }

  private static double durationToSeconds(Duration d) {
    return d.toNanos() / (double) TimeUnit.SECONDS.toNanos(1);
  }

  private static List<Entry<String, String>> mkCsvRow(
      List<Duration> xs, String name, int totalBytes, double megaScale, Options opt) {
    DoubleSummaryStatistics stats =
        xs.stream().skip(1).mapToDouble(Main::durationToSeconds).summaryStatistics();
    if (stats.getCount() <= 0) {
      throw new IllegalArgumentException("No min/max for " + name + "(size=" + totalBytes + ")");
    }
    double mbps = megaScale * (double) totalBytes / stats.getMin();
    return opt.csv
        ? Arrays.asList(
            new SimpleImmutableEntry<>("function", name),
            new SimpleImmutableEntry<>("num_times", opt.numtimes + ""),
            new SimpleImmutableEntry<>("n_elements", opt.arraysize + ""),
            new SimpleImmutableEntry<>("sizeof", totalBytes + ""),
            new SimpleImmutableEntry<>(
                "max_m" + (opt.mibibytes ? "i" : "") + "bytes_per_sec", mbps + ""),
            new SimpleImmutableEntry<>("min_runtime", stats.getMin() + ""),
            new SimpleImmutableEntry<>("max_runtime", stats.getMax() + ""),
            new SimpleImmutableEntry<>("avg_runtime", stats.getAverage() + ""))
        : Arrays.asList(
            new SimpleImmutableEntry<>("Function", name),
            new SimpleImmutableEntry<>(
                "M" + (opt.mibibytes ? "i" : "") + "Bytes/sec", String.format("%.3f", mbps)),
            new SimpleImmutableEntry<>("Min (sec)", String.format("%.5f", stats.getMin())),
            new SimpleImmutableEntry<>("Max", String.format("%.5f", stats.getMax())),
            new SimpleImmutableEntry<>("Average", String.format("%.5f", stats.getAverage())));
  }

  private static String padSpace(String s, int length) {
    if (length == 0) return s;
    return String.format("%1$-" + length + "s", s);
  }

  @SafeVarargs
  @SuppressWarnings("varargs")
  private static void tabulateCsv(boolean csv, List<Entry<String, String>>... rows) {
    if (rows.length == 0) throw new IllegalArgumentException("Empty tabulation");
    int padding = csv ? 0 : 12;
    String sep = csv ? "," : "";
    System.out.println(
        rows[0].stream().map(x -> padSpace(x.getKey(), padding)).collect(Collectors.joining(sep)));
    for (List<Entry<String, String>> row : rows) {
      System.out.println(
          row.stream().map(x -> padSpace(x.getValue(), padding)).collect(Collectors.joining(sep)));
    }
  }

  private static final String VERSION = "5.0";

  private static final float START_SCALAR = 0.4f;
  private static final float START_A = 0.1f;
  private static final float START_B = 0.2f;
  private static final float START_C = 0.0f;

  private static final List<Implementation> IMPLEMENTATIONS =
      Arrays.asList(
          new Implementation("jdk-stream", JdkStreams.FLOAT, JdkStreams.DOUBLE),
          new Implementation("jdk-plain", PlainStream.FLOAT, PlainStream.DOUBLE),
          new Implementation("tornadovm", TornadoVMStreams.FLOAT, TornadoVMStreams.DOUBLE),
          new Implementation("aparapi", AparapiStreams.FLOAT, AparapiStreams.DOUBLE));

  public static int run(String[] args) {
    Options opt = new Options();
    JCommander.newBuilder().addObject(opt).build().parse(args);

    final Benchmark benchmark;
    if (opt.nstreamOnly && opt.triadOnly)
      throw new RuntimeException(
          "Both triad and nstream are enabled, pick one or omit both to run all benchmarks");
    else if (opt.nstreamOnly) benchmark = Benchmark.NSTREAM;
    else if (opt.triadOnly) benchmark = Benchmark.TRIAD;
    else benchmark = Benchmark.ALL;

    final Config<Float> floatConfig =
        new Config<>(
            opt,
            benchmark,
            Float.BYTES,
            Float.class, // XXX not Float.TYPE, we want the boxed one
            Math.ulp(1.f),
            START_SCALAR,
            START_A,
            START_B,
            START_C);
    final Config<Double> doubleConfig =
        new Config<>(
            opt,
            benchmark,
            Double.BYTES,
            Double.class, // XXX not Double.TYPE, we want the boxed one
            Math.ulp(1.d),
            (double) START_SCALAR,
            (double) START_A,
            (double) START_B,
            (double) START_C);

    if (opt.list) {
      System.out.println("Set implementation with  --impl <IMPL> and device with --device <N>:");
      for (Implementation entry : IMPLEMENTATIONS) {
        System.out.println("Implementation: " + entry.name);
        try {
          List<String> devices = entry.makeDouble.apply(doubleConfig).listDevices();
          for (int i = 0; i < devices.size(); i++) {
            System.out.println("\t[" + i + "] " + devices.get(i));
          }
        } catch (Exception e) {
          System.out.println("\t(Unsupported: " + e.getMessage() + ")");
        }
      }
      return 0;
    }

    String implName = (opt.impl.isEmpty()) ? IMPLEMENTATIONS.get(0).name : opt.impl;
    Implementation impl =
        IMPLEMENTATIONS.stream()
            .filter(x -> implName.compareToIgnoreCase(x.name) == 0)
            .findFirst()
            .orElseThrow(
                () ->
                    new IllegalArgumentException("Implementation " + opt.impl + " does not exist"));

    boolean ok =
        opt.useFloat
            ? run(impl.name, floatConfig, impl.makeFloat)
            : run(impl.name, doubleConfig, impl.makeDouble);

    return ok ? 0 : 1;
  }

  public static void main(String[] args) {
    System.exit(run(args));
  }
}
