package scalastream

import scalastream.App.{Config, Data}

import scala.collection.immutable.ArraySeq
import scala.reflect.{ClassTag, classTag}

class J8SStream[@specialized(Float, Double) A: Fractional: ClassTag](val config: Config[A])
    extends ScalaStream[A]:

  private var a: Array[A] = _
  private var b: Array[A] = _
  private var c: Array[A] = _
  private val scalar: A   = config.scalar

  inline private def stream =
    java.util.stream.IntStream.range(0, config.options.arraysize).parallel()

  override inline def initArrays(): Unit =
    a = Array.ofDim(config.options.arraysize)
    b = Array.ofDim(config.options.arraysize)
    c = Array.ofDim(config.options.arraysize)
    stream.forEach { i =>
      a(i) = config.init._1
      b(i) = config.init._2
      c(i) = config.init._3
    }

  override inline def copy(): Unit    = stream.forEach(i => c(i) = a(i))
  override inline def mul(): Unit     = stream.forEach(i => b(i) = scalar * c(i))
  override inline def add(): Unit     = stream.forEach(i => c(i) = a(i) + b(i))
  override inline def triad(): Unit   = stream.forEach(i => a(i) = b(i) + scalar * c(i))
  override inline def nstream(): Unit = stream.forEach(i => a(i) = b(i) * scalar * c(i))
  override inline def dot(): A =
    // horrible special-case for double, there isn't a mapToFloat so we give up on that
    val cls = classTag[A].runtimeClass
    if java.lang.Double.TYPE == cls then
      stream
        .mapToDouble(i => (a(i) * b(i)).asInstanceOf[Double])
        .reduce(0, (l: Double, r: Double) => l + r)
        .asInstanceOf[A]
    else stream.mapToObj[A](i => a(i) * b(i)).reduce(0.fractional, (l: A, r: A) => l + r)

  override inline def data(): Data[A] = Data(a.to(ArraySeq), b.to(ArraySeq), c.to(ArraySeq))
