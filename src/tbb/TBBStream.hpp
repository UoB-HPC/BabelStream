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

#if defined(PARTITIONER_AUTO)
using tbb_partitioner = tbb::auto_partitioner;
#define PARTITIONER_NAME  "auto_partitioner"
#elif defined(PARTITIONER_AFFINITY)
using tbb_partitioner = tbb::affinity_partitioner;
#define PARTITIONER_NAME  "affinity_partitioner"
#elif defined(PARTITIONER_STATIC)
using tbb_partitioner = tbb::static_partitioner;
#define PARTITIONER_NAME  "static_partitioner"
#elif defined(PARTITIONER_SIMPLE)
using tbb_partitioner = tbb::simple_partitioner;
#define PARTITIONER_NAME  "simple_partitioner"
#else
// default to auto
using tbb_partitioner = tbb::auto_partitioner;
#define PARTITIONER_NAME  "auto_partitioner"
#endif


template <class T>
class TBBStream : public Stream<T>
{
  protected:
  
    tbb_partitioner partitioner;
    tbb::blocked_range<size_t> range;
    // Device side pointers
    std::vector<T> a;
    std::vector<T> b;
    std::vector<T> c;
 
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

