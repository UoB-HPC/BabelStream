#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#define IMPLEMENTATION_STRING "MUSA"

#define TBSIZE 1024

template <class T>
class MUSAStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;
    T *d_sum;

    // Number of blocks for dot kernel
    int dot_num_blocks;

  public:

    MUSAStream(const int, const int);
    ~MUSAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
