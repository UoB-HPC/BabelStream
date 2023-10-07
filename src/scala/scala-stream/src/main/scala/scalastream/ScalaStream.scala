package scalastream
import scalastream.App.{Config, Data, Timings}

import java.util.concurrent.TimeUnit
import scala.collection.immutable.ArraySeq
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration.{Duration, FiniteDuration, SECONDS}
import scala.math.{Pi, pow}
import scala.reflect.ClassTag
import scopt.OParser

transparent trait ScalaStream[@specialized(Float, Double) A]:

  def config: Config[A]

  def initArrays(): Unit
  def readArrays(): Unit = ()
  def copy(): Unit
  def mul(): Unit
  def add(): Unit
  def triad(): Unit
  def nstream(): Unit
  def dot(): A

  transparent inline def timed[R](f: => R): (FiniteDuration, R) =
    val start = System.nanoTime()
    val r     = f
    val end   = System.nanoTime()
    FiniteDuration(end - start, TimeUnit.NANOSECONDS) -> r

  inline def runInitArrays(): FiniteDuration = timed(initArrays())._1
  inline def runReadArrays(): FiniteDuration = timed(readArrays())._1
  inline def runAll(times: Int)(using Fractional[A]): (Timings[Vector[FiniteDuration]], A) =
    val copy  = ArrayBuffer.fill[FiniteDuration](times)(Duration.Zero)
    val mul   = ArrayBuffer.fill[FiniteDuration](times)(Duration.Zero)
    val add   = ArrayBuffer.fill[FiniteDuration](times)(Duration.Zero)
    val triad = ArrayBuffer.fill[FiniteDuration](times)(Duration.Zero)
    val dot   = ArrayBuffer.fill[FiniteDuration](times)(Duration.Zero)

    var lastSum: A = 0.fractional

    for i <- 0 until times do
      copy(i) = timed(this.copy())._1
      mul(i) = timed(this.mul())._1
      add(i) = timed(this.add())._1
      triad(i) = timed(this.triad())._1
      val (dot_, sum) = timed(this.dot())
      dot(i) = dot_
      lastSum = sum
    val s = lastSum

    (
      Timings(
        copy = copy.toVector,
        mul = mul.toVector,
        add = add.toVector,
        triad = triad.toVector,
        dot = dot.toVector
      ),
      s
    )

  def runTriad(times: Int): FiniteDuration           = timed(for _ <- 0 until times do triad())._1
  def runNStream(times: Int): Vector[FiniteDuration] = Vector.fill(times)(timed(nstream())._1)

  def data(): Data[A]

trait Fractional[@specialized(Double, Float) A]:
  def toFractional(f: Float): A
  def toFractional(f: Double): A
  def compare(x: A, y: A): Int
  def add(x: A, y: A): A
  def sub(x: A, y: A): A
  def mul(x: A, y: A): A
  def div(x: A, y: A): A
  def abs(x: A): A
  extension (x: Float) inline def fractional  = toFractional(x)
  extension (x: Double) inline def fractional = toFractional(x)
  extension (x: Int) inline def fractional    = toFractional(x.toFloat)
  extension (x: Long) inline def fractional   = toFractional(x.toDouble)
  extension (x: A)
    inline def +(y: A) = add(x, y)
    inline def -(y: A) = sub(x, y)
    inline def *(y: A) = mul(x, y)
    inline def /(y: A) = div(x, y)
    inline def >(y: A) = compare(x, y) > 0
    inline def <(y: A) = compare(x, y) < 0
    inline def abs_    = abs(x)
end Fractional

given FloatFractional: Fractional[Float] with
  inline def toFractional(f: Float): Float    = f
  inline def toFractional(f: Double): Float   = f.toFloat
  inline def compare(x: Float, y: Float): Int = x.compare(y)
  inline def add(x: Float, y: Float): Float   = x + y
  inline def sub(x: Float, y: Float): Float   = x - y
  inline def mul(x: Float, y: Float): Float   = x * y
  inline def div(x: Float, y: Float): Float   = x / y
  inline def abs(x: Float): Float             = math.abs(x)

given DoubleFractional: Fractional[Double] with
  inline def toFractional(f: Float): Double     = f.toDouble
  inline def toFractional(f: Double): Double    = f
  inline def compare(x: Double, y: Double): Int = x.compare(y)
  inline def add(x: Double, y: Double): Double  = x + y
  inline def sub(x: Double, y: Double): Double  = x - y
  inline def mul(x: Double, y: Double): Double  = x * y
  inline def div(x: Double, y: Double): Double  = x / y
  inline def abs(x: Double): Double             = math.abs(x)

object App:

  final val Version: String = "5.0"

  case class Config[@specialized(Double, Float) A](
      options: Options,
      benchmark: Benchmark,
      typeSize: Int,
      ulp: A,
      scalar: A,
      init: (A, A, A)
  )

  case class Timings[A](copy: A, mul: A, add: A, triad: A, dot: A)
  case class Data[A](@specialized(Double, Float) a: ArraySeq[A], b: ArraySeq[A], c: ArraySeq[A])

  case class Options(
      list: Boolean = false,
      device: Int = 0,
      numtimes: Int = 100,
      arraysize: Int = 33554432,
      float: Boolean = false,
      triad_only: Boolean = false,
      nstream_only: Boolean = false,
      csv: Boolean = false,
      mibibytes: Boolean = false
  )

  object Options:
    val Default = Options()
    val builder = OParser.builder[Options]
    val parser1 =
      import builder._
      OParser.sequence(
        programName("scala-stream"),
        head("ScalaStream", s"$Version"),
        opt[Unit]('l', "list").text("List available devices").action((_, x) => x.copy(list = true)),
        opt[Int]('d', "device")
          .text(s"Select device at <device>, defaults to ${Default.device}")
          .action((v, x) => x.copy(device = v)),
        opt[Int]('n', "numtimes")
          .text(s"Run the test <numtimes> times (NUM >= 2), defaults to ${Default.numtimes}")
          .validate {
            case n if n >= 2 => success
            case n           => failure(s"$n <= 2")
          }
          .action((n, x) => x.copy(numtimes = n)),
        opt[Int]('a', "arraysize")
          .text(s"Use <arraysize> elements in the array, defaults to ${Default.arraysize}")
          .action((v, x) => x.copy(arraysize = v)),
        opt[Unit]('f', "float")
          .text("Use floats (rather than doubles)")
          .action((_, x) => x.copy(float = true)),
        opt[Unit]('t', "triad_only")
          .text("Only run triad")
          .action((_, x) => x.copy(triad_only = true)),
        opt[Unit]('n', "nstream_only")
          .text("Only run nstream")
          .action((_, x) => x.copy(nstream_only = true)),
        opt[Unit]('c', "csv").text("Output as csv table").action((_, x) => x.copy(csv = true)),
        opt[Unit]('m', "mibibytes")
          .text("Use MiB=2^20 for bandwidth calculation (default MB=10^6)")
          .action((_, x) => x.copy(mibibytes = true)),
        help('h', "help").text("prints this usage text")
      )

  enum Benchmark:
    case All, NStream, Triad

  implicit class RichDuration(private val d: Duration) extends AnyVal:
    def seconds: Double = d.toUnit(SECONDS)

  def validate[A: Fractional](vec: Data[A], config: Config[A], dotSum: Option[A] = None): Unit =

    var (goldA, goldB, goldC) = config.init
    for _ <- 0 until config.options.numtimes do
      config.benchmark match
        case Benchmark.All =>
          goldC = goldA
          goldB = config.scalar * goldC
          goldC = goldA + goldB
          goldA = goldB + config.scalar * goldC
        case Benchmark.Triad =>
          goldA = goldB + config.scalar * goldC
        case Benchmark.NStream =>
          goldA += goldB + config.scalar * goldC

    val tolerance = config.ulp * (100.fractional)
    def validateXs(name: String, xs: Seq[A], from: A): Unit =
      val error = xs.map(x => (x - from).abs_).fold(0.fractional)(_ + _) / xs.size.fractional
      if error > tolerance then
        Console.err.println(s"Validation failed on $name. Average error $error ")

    validateXs("a", vec.a, goldA)
    validateXs("b", vec.b, goldB)
    validateXs("c", vec.c, goldC)

    dotSum.foreach { sum =>
      val goldSum = (goldA * goldB) * config.options.arraysize.fractional
      val error   = ((sum - goldSum) / goldSum).abs_
      if error > 1.fractional / 100000000.fractional then
        Console.err.println(
          s"Validation failed on sum. Error $error \nSum was $sum but should be $goldSum"
        )
    }

  inline def run[A: Fractional: ClassTag](
      name: String,
      config: Config[A],
      mkStream: Config[A] => ScalaStream[A]
  ): Unit =

    val opt = config.options

    val arrayBytes = opt.arraysize * config.typeSize
    val totalBytes = arrayBytes * 3
    val (megaScale, megaSuffix, gigaScale, gigaSuffix) =
      if !opt.mibibytes then (1.0e-6, "MB", 1.0e-9, "GB")
      else (pow(2.0, -20), "MiB", pow(2.0, -30), "GiB")

    if !opt.csv then

      val vendor = System.getProperty("java.vendor")
      val ver    = System.getProperty("java.version")
      val home   = System.getProperty("java.home")
      println(
        s"""BabelStream
           |Version: $Version
           |Implementation: $name; Scala (Java $ver; $vendor; home=$home)""".stripMargin
      )

      println(s"Running ${config.benchmark match {
          case Benchmark.All     => "kernels"
          case Benchmark.Triad   => "triad"
          case Benchmark.NStream => "nstream"
        }} ${opt.numtimes} times")

      if config.benchmark == Benchmark.Triad then println(s"Number of elements: ${opt.arraysize}")

      println(s"Precision: ${if opt.float then "float" else "double"}")
      println(
        f"Array size: ${megaScale * arrayBytes}%.1f $megaSuffix (=${gigaScale * arrayBytes}%.1f $gigaSuffix)"
      )
      println(
        f"Total size: ${megaScale * totalBytes}%.1f $megaSuffix (=${gigaScale * totalBytes}%.1f $gigaSuffix)"
      )

    def mkRow(xs: Vector[FiniteDuration], name: String, totalBytes: Int) =
      val tail = xs.tail
      (tail.minOption.map(_.seconds), tail.maxOption.map(_.seconds)) match
        case (Some(min), Some(max)) =>
          val avg  = (tail.foldLeft(Duration.Zero)(_ + _) / tail.size.toDouble).seconds
          val mbps = megaScale * totalBytes.toDouble / min
          if opt.csv then
            Vector(
              "function"                                                -> name,
              "num_times"                                               -> opt.numtimes.toString,
              "n_elements"                                              -> opt.arraysize.toString,
              "sizeof"                                                  -> totalBytes.toString,
              s"max_m${if opt.mibibytes then "i" else ""}bytes_per_sec" -> mbps.toString,
              "min_runtime"                                             -> min.toString,
              "max_runtime"                                             -> max.toString,
              "avg_runtime"                                             -> avg.toString
            )
          else
            Vector(
              "Function"                                        -> name,
              s"M${if opt.mibibytes then "i" else ""}Bytes/sec" -> f"$mbps%.3f",
              "Min (sec)"                                       -> f"$min%.5f",
              "Max"                                             -> f"$max%.5f",
              "Average"                                         -> f"$avg%.5f"
            )
        case (_, _) => sys.error(s"No min/max element for $name(size=$totalBytes)")

    def tabulate(rows: Vector[(String, String)]*): Unit = rows.toList match
      case Nil => sys.error(s"Empty tabulation")
      case header :: _ =>
        val padding = if opt.csv then 0 else 12
        val sep     = if opt.csv then "," else ""
        println(header.map(_._1.padTo(padding, ' ')).mkString(sep))
        println(rows.map(_.map(_._2.padTo(padding, ' ')).mkString(sep)).mkString("\n"))

    def showInit(init: FiniteDuration, read: FiniteDuration): Unit = {
      val setup =
        Vector(("Init", init.seconds, 3 * arrayBytes), ("Read", read.seconds, 3 * arrayBytes))
      if opt.csv then
        tabulate(
          setup.map((name, elapsed, totalBytes) =>
            Vector(
              "phase"      -> name,
              "n_elements" -> opt.arraysize.toString,
              "sizeof"     -> arrayBytes.toString,
              s"max_m${if opt.mibibytes then "i" else ""}bytes_per_sec" ->
                (megaScale * totalBytes.toDouble / elapsed).toString,
              "runtime" -> elapsed.toString
            )
          ): _*
        )
      else
        for (name, elapsed, totalBytes) <- setup do
          println(
            f"$name: $elapsed%.5f s (=${megaScale * totalBytes.toDouble / elapsed}%.5f M${
                if opt.mibibytes then "i" else ""
              }Bytes/sec)"
          )
    }

    val stream = mkStream(config)
    val init   = stream.runInitArrays()
    config.benchmark match
      case Benchmark.All =>
        val (results, sum) = stream.runAll(opt.numtimes)
        val read           = stream.runReadArrays()
        showInit(init, read)
        validate(stream.data(), config, Some(sum))
        tabulate(
          mkRow(results.copy, "Copy", 2 * arrayBytes),
          mkRow(results.mul, "Mul", 2 * arrayBytes),
          mkRow(results.add, "Add", 3 * arrayBytes),
          mkRow(results.triad, "Triad", 3 * arrayBytes),
          mkRow(results.dot, "Dot", 2 * arrayBytes)
        )
      case Benchmark.NStream =>
        val result = stream.runNStream(opt.numtimes)
        val read   = stream.runReadArrays()
        showInit(init, read)
        validate(stream.data(), config)
        tabulate(mkRow(result, "Nstream", 4 * arrayBytes))
      case Benchmark.Triad =>
        val results = stream.runTriad(opt.numtimes)
        val read    = stream.runReadArrays()
        showInit(init, read)
        val totalBytes = 3 * arrayBytes * opt.numtimes
        val bandwidth  = megaScale * (totalBytes / results.seconds)
        println(f"Runtime (seconds): ${results.seconds}%.5f")
        println(f"Bandwidth ($gigaSuffix/s): $bandwidth%.3f ")

  inline def devices[A: Fractional: ClassTag]: Vector[(String, Config[A] => ScalaStream[A])] =
    Vector(
      "Scala Parallel Collections" -> (ParStream(_)),
      "Java 8 Stream"              -> (J8SStream(_)),
      "Threaded"                   -> (ThreadStream(_)),
      "Serial"                     -> (PlainStream(_))
    )

  inline def runWith[A: Fractional: ClassTag](i: Int, config: Config[A]): Unit =
    devices[A].lift(i) match
      case None                   => println(s"Device index out of bounds: $i")
      case Some((name, mkStream)) => run(name, config, mkStream)

  def main(args: Array[String]): Unit =

    def handleOpt(opt: Options) =
      val benchmark = (opt.nstream_only, opt.triad_only) match
        case (true, false)  => Benchmark.NStream
        case (false, true)  => Benchmark.Triad
        case (false, false) => Benchmark.All
        case (true, true) =>
          throw new RuntimeException(
            "Both triad and nstream are enabled, pick one or omit both to run all benchmarks"
          )

      if opt.list then
        devices[Float].zipWithIndex.foreach { case ((name, _), i) => println(s"$i: $name") }
      else if opt.float then
        runWith(
          opt.device,
          Config(
            options = opt,
            benchmark = benchmark,
            typeSize = 4, // 32bit
            ulp = math.ulp(Float.MaxValue),
            scalar = 0.4f,
            init = (0.1f, 0.2f, 0.0f)
          )
        )
      else
        runWith(
          opt.device,
          Config(
            options = opt,
            benchmark = benchmark,
            typeSize = 8,
            ulp = math.ulp(Double.MaxValue),
            scalar = 0.4, // 64bit
            init = (0.1, 0.2, 0.0)
          )
        )

    OParser.parse(Options.parser1, args, Options.Default) match
      case Some(config) => handleOpt(config)
      case _            => sys.exit(1)
