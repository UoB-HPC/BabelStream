

#include "Stream.h"

template <class T>
class CUDAStream : public Stream<T>
{
  public:
    void copy();
    void add();
    void mul();
    void triad();
};
