
#include <iostream>
#include <stdexcept>

#include "Stream.h"

template <class T>
class CUDAStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;
    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;


  public:

    CUDAStream(const unsigned int);
    ~CUDAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

