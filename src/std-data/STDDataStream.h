// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include "Stream.h"

#define IMPLEMENTATION_STRING "STD (data-oriented)"

#ifdef USE_ONEDPL
    #define PSTL_USAGE_WARNINGS 1

    #include <oneapi/dpl/execution>
    #include <oneapi/dpl/iterator>
    #include <oneapi/dpl/algorithm>
    #include <oneapi/dpl/memory>
    #include <oneapi/dpl/numeric>

    #ifdef ONEDPL_USE_DPCPP_BACKEND
        #include <CL/sycl.hpp>
    #endif
#else
    #include <algorithm>
    #include <execution>
    #include <numeric>

    #if defined(ONEDPL_USE_DPCPP_BACKEND) || \
        defined(ONEDPL_USE_TBB_BACKEND)   || \
        defined(ONEDPL_USE_OPENMP_BACKEND)
        #error oneDPL missing (ONEDPL_VERSION_MAJOR not defined) but backend (ONEDPL_USE_*_BACKEND) specified
    #endif

#endif


template <class T>
class STDDataStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

#if defined(ONEDPL_USE_DPCPP_BACKEND)
    // SYCL oneDPL backend
    using ExecutionPolicy = oneapi::dpl::execution::device_policy<>;
#elif defined(USE_ONEDPL)
    // every other non-SYCL oneDPL backend (i.e TBB, OMP)
    using ExecutionPolicy = decltype(oneapi::dpl::execution::par_unseq);
#else
    // normal std execution policies
    using ExecutionPolicy = decltype(std::execution::par_unseq);
#endif

    ExecutionPolicy exe_policy{};

    // Device side pointers
    T* a;
    T* b;
    T* c;


  public:
    STDDataStream(const int, int);
    ~STDDataStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

