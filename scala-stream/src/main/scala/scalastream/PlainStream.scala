package scalastream

import scalastream.App.{Config, Data}

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
class PlainStream[@specialized(Float, Double) A: Fractional: ClassTag](val config: Config[A])
    extends ScalaStream[A]:

  private var a: Array[A] = _
  private var b: Array[A] = _
  private var c: Array[A] = _
  private val scalar: A   = config.scalar

  override inline def initArrays(): Unit =
    a = Array.fill(config.options.arraysize)(config.init._1)
    b = Array.fill(config.options.arraysize)(config.init._2)
    c = Array.fill(config.options.arraysize)(config.init._3)

  private inline def indices = 0 until config.options.arraysize

  override inline def copy(): Unit    = for i <- indices do c(i) = a(i)
  override inline def mul(): Unit     = for i <- indices do b(i) = scalar * c(i)
  override inline def add(): Unit     = for i <- indices do c(i) = a(i) + b(i)
  override inline def triad(): Unit   = for i <- indices do a(i) = b(i) + (scalar * c(i))
  override inline def nstream(): Unit = for i <- indices do a(i) = b(i) * scalar * c(i)
  override inline def dot(): A =
    var acc: A = 0.fractional
    for i <- indices do acc = acc + (a(i) * b(i))
    acc
  override inline def data(): Data[A] = Data(a.to(ArraySeq), b.to(ArraySeq), c.to(ArraySeq))
