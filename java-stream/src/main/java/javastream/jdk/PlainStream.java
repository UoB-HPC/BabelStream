package javastream.jdk;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.function.Function;
import javastream.JavaStream;
import javastream.JavaStream.EnumeratedStream;
import javastream.Main.Config;

public final class PlainStream {

  private PlainStream() {}

  public static final Function<Config<Float>, JavaStream<Float>> FLOAT =
      config ->
          new EnumeratedStream<>(
              config,
              new SimpleImmutableEntry<>("specialised", SpecialisedPlainFloatStream::new),
              new SimpleImmutableEntry<>("generic", GenericPlainStream::new));

  public static final Function<Config<Double>, JavaStream<Double>> DOUBLE =
      config ->
          new EnumeratedStream<>(
              config,
              new SimpleImmutableEntry<>("specialised", SpecialisedPlainDoubleStream::new),
              new SimpleImmutableEntry<>("generic", GenericPlainStream::new));
}
