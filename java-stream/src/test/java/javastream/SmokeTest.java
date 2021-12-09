package javastream;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

public class SmokeTest {

  // taken from https://stackoverflow.com/a/32146095/896997
  private static <T> Stream<List<T>> ofCombinations(
      List<? extends Collection<T>> collections, List<T> current) {
    return collections.isEmpty()
        ? Stream.of(current)
        : collections.get(0).stream()
            .flatMap(
                e -> {
                  List<T> list = new ArrayList<>(current);
                  list.add(e);
                  return ofCombinations(collections.subList(1, collections.size()), list);
                });
  }

  @SuppressWarnings("unused")
  private static Stream<Arguments> options() {

    LinkedHashMap<String, List<Integer>> impls = new LinkedHashMap<>();
    impls.put("jdk-stream", Arrays.asList(0, 1));
    impls.put("jdk-plain", Arrays.asList(0, 1));
    // skip aparapi as none of the jdk fallbacks work correctly
    // skip tornadovm as it has no jdk fallback

    List<String> configs =
        impls.entrySet().stream()
            .flatMap(
                e ->
                    Stream.concat(Stream.of(""), e.getValue().stream().map(i -> "--device " + i))
                        .map(d -> "--impl " + e.getKey() + " " + d))
            .collect(Collectors.toList());

    return ofCombinations(
            new ArrayList<>(
                Arrays.asList(
                    configs,
                    Arrays.asList("", "--csv"),
                    // XXX floats usually have a 1.0^-5 error which misses 10^-8
                    Arrays.asList("", "--float --dot-tolerance 1.0e-5"),
                    Arrays.asList("", "--triad-only", "--nstream-only"),
                    Arrays.asList("", "--mibibytes"))),
            Collections.emptyList())
        .map(
            xs ->
                Arguments.of(
                    xs.stream() //
                        .map(String::trim) //
                        .collect(Collectors.joining(" "))
                        .trim()));
  }

  @ParameterizedTest
  @MethodSource("options")
  void testIt(String args) {
    String line = "--arraysize 2048 " + args;

    // redirect stdout/stderr and only print if anything fails
    ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    ByteArrayOutputStream errContent = new ByteArrayOutputStream();
    PrintStream originalOut = System.out;
    PrintStream originalErr = System.err;

    System.setOut(new PrintStream(outContent));
    System.setErr(new PrintStream(errContent));
    int run = Main.run(line.split("\\s+"));
    System.setOut(originalOut);
    System.setErr(originalErr);

    if (run != 0) {
      System.out.println(outContent);
      System.err.println(errContent);
      Assertions.assertEquals(0, run, "`" + line + "` did not return 0");
    }
  }
}
