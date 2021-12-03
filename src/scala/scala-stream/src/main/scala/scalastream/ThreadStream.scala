package scalastream

import net.openhft.affinity.{AffinityStrategies, AffinityThreadFactory}
import scalastream.App.{Config, Data}

import java.util.concurrent.{Callable, Executors}
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
object ThreadStream {}
class ThreadStream[@specialized(Float, Double) A: Fractional: ClassTag](val config: Config[A])
    extends ScalaStream[A]:

  private var a: Array[A] = _
  private var b: Array[A] = _
  private var c: Array[A] = _
  private val scalar: A   = config.scalar

  private val chunks: Int = sys.runtime.availableProcessors()

  private val pool = Executors.newFixedThreadPool(
    chunks,
    new AffinityThreadFactory("scala-stream", true, AffinityStrategies.DIFFERENT_CORE)
  )

  private val indices = (0 until config.options.arraysize)
    .grouped(config.options.arraysize / chunks)
    .toSeq

  private inline def forEachAll[C](c: => C)(f: (C, Int) => Unit): Seq[C] =
    import scala.jdk.CollectionConverters._
    val xs = pool
      .invokeAll(
        indices.map { r =>
          { () =>
            val ctx = c
            r.foreach(f(ctx, _))
            ctx
          }: Callable[C]
        }.asJavaCollection
      )
      .asScala
      .map(_.get())
      .toSeq
    xs

  override inline def initArrays(): Unit =
    a = Array.ofDim(config.options.arraysize)
    b = Array.ofDim(config.options.arraysize)
    c = Array.ofDim(config.options.arraysize)
    forEachAll(()) { (_, i) =>
      a(i) = config.init._1
      b(i) = config.init._2
      c(i) = config.init._3
    }
    ()

  class Box(var value: A)
  override inline def copy(): Unit = { forEachAll(())((_, i) => c(i) = a(i)); () }
  override inline def mul(): Unit = { forEachAll(())((_, i) => b(i) = scalar * c(i)); () }
  override inline def add(): Unit = { forEachAll(())((_, i) => c(i) = a(i) + b(i)); () }
  override inline def triad(): Unit = { forEachAll(())((_, i) => a(i) = b(i) + scalar * c(i)); () }
  override inline def nstream(): Unit = { forEachAll(())((_, i) => a(i) = b(i) * scalar * c(i)); () }

  override inline def dot(): A =
    forEachAll(Box(0.fractional))((acc, i) => acc.value = acc.value + (a(i) * b(i)))
      .map(_.value)
      .fold(0.fractional)(_ + _)
  override inline def data(): Data[A] = Data(a.to(ArraySeq), b.to(ArraySeq), c.to(ArraySeq))
