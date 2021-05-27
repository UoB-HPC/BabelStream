// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <vector>
#include "tbb/tbb.h"
#include "Stream.h"

#define IMPLEMENTATION_STRING "TBB"

enum class Partitioner : int { Auto = 0, Affinity, Static, Simple};

template <class T>
class TBBStream : public Stream<T>
{
  protected:
  

    Partitioner partitioner;
    tbb::blocked_range<size_t> range;
    // Device side pointers
    std::vector<T> a;
    std::vector<T> b;
    std::vector<T> c;
    

    template < typename U, typename F>
    U with_partitioner(const F &f);
 
    template <typename F>
    void parallel_for(const F &f);

    template <typename F, typename Op>
    T parallel_reduce(T init, const Op &op, const F &f);

  public:
    TBBStream(const int, int);
    ~TBBStream() = default;

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

