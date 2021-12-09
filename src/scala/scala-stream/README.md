ScalaStream
===========

This is an implementation of BabelStream
in [Scala 3](https://docs.scala-lang.org/scala3/new-in-scala3.html) on the JVM. In theory, this
implementation also covers Java. Scala and Java, like any other programming language, has its own
ecosystem of library supported parallel programming frameworks, we currently implement the
following:

* Parallel streams (introduced in Java 8) - `src/main/scala/scalastream/J8SStream.scala`
* [Scala Parallel Collections](https://github.com/scala/scala-parallel-collections)
    - `src/main/scala/scalastream/ParStream.scala`

As the benchmark is relatively simple, we also implement some baselines:

* Single threaded Scala `for` (i.e `foreach` sugar) - `src/main/scala/scalastream/PlainStream.scala`
* Manually parallelism with Java executors - `src/main/scala/scalastream/ThreadedStream.scala`

### Performance considerations

As Scala 3 defaults to Scala 2.13's standard library, we roll our own `Fractional` typeclass with
liberal use of inlining and specialisation. This is motivated by 2.13 stdlib's lack of
specialisation for primitives types on the default `Fractional` and `Numeric` typeclasses.

The use of [Spire](https://github.com/typelevel/spire) to mitigate this was attempted, however, due
to its use of Scala 2 macros, it currently doesn't compile with Scala 3.

### Build & Run

Prerequisites

* JDK >= 8 on any of its supported platform; known working implementations:
    - OpenJDK
      distributions ([Amazon Corretto](https://docs.aws.amazon.com/corretto/latest/corretto-11-ug/downloads-list.html)
      , [Azul](https://www.azul.com/downloads/?version=java-11-lts&package=jdk)
      , [AdoptOpenJDK](https://adoptopenjdk.net/), etc)
    - Oracle Graal CE/EE 8+

To run the benchmark, first create a binary:

```shell
> ./sbt assembly
```

The binary will be located at `./target/scala-3.0.0/scala-stream.jar`. Run it with:

```shell
> java -version 
openjdk version "11.0.11" 2021-04-20
OpenJDK Runtime Environment 18.9 (build 11.0.11+9)
OpenJDK 64-Bit Server VM 18.9 (build 11.0.11+9, mixed mode, sharing)
> java  -jar target/scala-3.0.0/scala-stream.jar --help

```

For best results, benchmark with the following JVM flags:

```
-XX:-UseOnStackReplacement     # disable OSR, not useful for this benchmark as we are measuring peak performance  
-XX:-TieredCompilation         # disable C1, go straight to C2 
-XX:ReservedCodeCacheSize=512m # don't flush compiled code out of cache at any point 
```

Worked example:

```shell
> java -XX:-UseOnStackReplacement -XX:-TieredCompilation -XX:ReservedCodeCacheSize=512m -jar target/scala-3.0.0/scala-stream.jar

BabelStream
Version: 3.4.0
Implementation: Scala Parallel Collections; Scala (Java 11.0.11; Red Hat, Inc.; home=/usr/lib/jvm/java-11-openjdk-11.0.11.0.9-2.fc33.x86_64)
Running kernels 100 times
Precision: double
Array size: 268.4 MB (=0.3 GB)
Total size: 805.3 MB (=0.8 GB)
Function    MBytes/sec  Min (sec)   Max         Average     
Copy        4087.077    0.13136     0.24896     0.15480     
Mul         2934.709    0.18294     0.28706     0.21627     
Add         3016.342    0.26698     0.39835     0.31119     
Triad       3016.496    0.26697     0.37612     0.31040     
Dot         2216.096    0.24226     0.41235     0.28264

```

### Graal Native Image

The port has partial support for Graal Native Image, to generate one, run:

```shell
> ./sbt nativeImage
```

The ELF binary will be located at `./target/native-image/scala-stream`, relocation should work on
the same architecture the binary is built on.

There's an ongoing bug with Scala 3 's use of `lazy val`s where the program crashes at declaration
site. Currently, Scala Parallel Collections uses this feature internally, so selecting this device
will crash at runtime.

The bug originates from the use of  `Unsafe` in `lazy val` for thready safety guarantees. It seems
that Graal only supports limited uses of this JVM implementation detail and Scala 3 happens to be on
the unsupported side.  