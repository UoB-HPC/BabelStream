package javastream.aparapi;

import com.aparapi.Kernel;
import com.aparapi.Range;
import javastream.JavaStream.Data;
import javastream.Main.Config;

abstract class GenericAparapiStreamKernel<T> extends Kernel {

  protected static final int FN_COPY = 1;
  protected static final int FN_MUL = 2;
  protected static final int FN_ADD = 3;
  protected static final int FN_TRIAD = 4;
  protected static final int FN_NSTREAM = 5;
  protected static final int FN_DOT = 6;
  protected final Config<T> config;
  protected final int arraysize, numGroups, workGroupSize;

  interface Factory<T> {
    GenericAparapiStreamKernel<T> create(Config<T> config, int numGroups, int workGroupSize);
  }

  GenericAparapiStreamKernel(Config<T> config, int numGroups, int workGroupSize) {
    this.config = config;
    this.arraysize = config.options.arraysize;
    this.numGroups = numGroups;
    this.workGroupSize = workGroupSize;
    setExplicit(true);
  }

  protected int function;

  public abstract void init();

  public void copy() {
    function = FN_COPY;
    execute(arraysize);
  }

  public void mul() {
    function = FN_MUL;
    execute(arraysize);
  }

  public void add() {
    function = FN_ADD;
    execute(arraysize);
  }

  public void triad() {
    function = FN_TRIAD;
    execute(arraysize);
  }

  public void nstream() {
    function = FN_NSTREAM;
    execute(arraysize);
  }

  protected Kernel partialDot() {
    function = FN_DOT;
    return execute(Range.create(numGroups * workGroupSize, workGroupSize));
  }

  abstract T dot();

  abstract Data<T> syncAndDispose();
}
