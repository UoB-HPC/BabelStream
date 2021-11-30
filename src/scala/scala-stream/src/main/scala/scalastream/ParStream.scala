package scalastream

import scalastream.App.{Config, Data}

import scala.collection.immutable.ArraySeq
import scala.collection.parallel.CollectionConverters._
import scala.reflect.ClassTag
class ParStream[@specialized(Float, Double) A: Fractional: ClassTag](val config: Config[A])
    extends ScalaStream[A]:

  private var a: Array[A] = _
  private var b: Array[A] = _
  private var c: Array[A] = _
  private val scalar: A   = config.scalar

  inline private def indices = (0 until config.options.arraysize).par

  override inline def initArrays(): Unit =
    a = Array.ofDim(config.options.arraysize)
    b = Array.ofDim(config.options.arraysize)
    c = Array.ofDim(config.options.arraysize)

    for i <- indices do
      a(i) = config.init._1
      b(i) = config.init._2
      c(i) = config.init._3

  override inline def copy(): Unit    = for i <- indices do c(i) = a(i)
  override inline def mul(): Unit     = for i <- indices do b(i) = scalar * c(i)
  override inline def add(): Unit     = for i <- indices do c(i) = a(i) + b(i)
  override inline def triad(): Unit   = for i <- indices do a(i) = b(i) + scalar * c(i)
  override inline def nstream(): Unit = for i <- indices do a(i) = b(i) * scalar * c(i)
  override inline def dot(): A =
    indices.aggregate[A](0.fractional)((acc, i) => acc + (a(i) * b(i)), _ + _)

  override inline def data(): Data[A] = Data(a.to(ArraySeq), b.to(ArraySeq), c.to(ArraySeq))
