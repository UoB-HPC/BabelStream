package javastream;

/**
 * This class represents our Fractional typeclass. Java's type system isn't unified so we have to do
 * insane things for parametric operations on fractional types.
 */
@SuppressWarnings("unchecked")
public final class FractionalMaths {

  private FractionalMaths() {
    throw new AssertionError();
  }

  public static <T extends Number> T from(Class<T> evidence, Number n) {
    if (evidence == Double.TYPE || evidence == Double.class)
      return (T) Double.valueOf(n.doubleValue());
    else if (evidence == Float.TYPE || evidence == Float.class)
      return (T) Float.valueOf(n.floatValue());
    throw new IllegalArgumentException();
  }

  public static <T extends Number> T plus(T x, T y) {
    if (x instanceof Double) return (T) Double.valueOf(x.doubleValue() + y.doubleValue());
    else if (x instanceof Float) return (T) Float.valueOf(x.floatValue() + y.floatValue());
    throw new IllegalArgumentException();
  }

  static <T extends Number> T minus(T x, T y) {
    if (x instanceof Double) return (T) Double.valueOf(x.doubleValue() - y.doubleValue());
    else if (x instanceof Float) return (T) Float.valueOf(x.floatValue() - y.floatValue());
    throw new IllegalArgumentException();
  }

  public static <T extends Number> T times(T x, T y) {
    if (x instanceof Double) return (T) Double.valueOf(x.doubleValue() * y.doubleValue());
    else if (x instanceof Float) return (T) Float.valueOf(x.floatValue() * y.floatValue());
    throw new IllegalArgumentException();
  }

  static <T extends Number> T divide(T x, T y) {
    if (x instanceof Double) return (T) Double.valueOf(x.doubleValue() / y.doubleValue());
    else if (x instanceof Float) return (T) Float.valueOf(x.floatValue() / y.floatValue());
    throw new IllegalArgumentException();
  }
}
