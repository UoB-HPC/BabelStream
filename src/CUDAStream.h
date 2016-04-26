

#include "Stream.h"

template <class T>
class CUDAStream : public Stream<T>
{
  public:
    void copy();
    void add();
    void mul();
    void triad();

  private:
    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;

};
