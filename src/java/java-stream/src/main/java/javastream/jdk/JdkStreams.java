package javastream.jdk;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.function.Function;
import javastream.JavaStream;
import javastream.JavaStream.EnumeratedStream;
import javastream.Main.Config;

public final class JdkStreams {

  private JdkStreams() {}

  public static final Function<Config<Float>, JavaStream<Float>> FLOAT =
      config ->
          new EnumeratedStream<>(
              config,
              new SimpleImmutableEntry<>("specialised", SpecialisedFloatStream::new),
              new SimpleImmutableEntry<>("generic", GenericStream::new));

  public static final Function<Config<Double>, JavaStream<Double>> DOUBLE =
      config ->
          new EnumeratedStream<>(
              config,
              new SimpleImmutableEntry<>("specialised", SpecialisedDoubleStream::new),
              new SimpleImmutableEntry<>("generic", GenericStream::new));
}
