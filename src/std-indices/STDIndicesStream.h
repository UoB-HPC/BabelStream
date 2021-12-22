// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include "Stream.h"

#define IMPLEMENTATION_STRING "STD (index-oriented)"



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

// A lightweight counting iterator which will be used by the STL algorithms
// NB: C++ <= 17 doesn't have this built-in, and it's only added later in ranges-v3 (C++2a) which this
// implementation doesn't target
template <typename N>
class ranged_iterator {
  N num;
public:
  using difference_type = N;
  using value_type = N;
  using pointer = const N*;
  using reference = const N&;
  using iterator_category = std::random_access_iterator_tag;
  explicit ranged_iterator(N _num = 0) : num(_num) {}

  ranged_iterator<N>& operator++() { num++; return *this; }
  ranged_iterator<N> operator++(int) { ranged_iterator<N> retval = *this; ++(*this); return retval; }
  ranged_iterator<N> operator+(const value_type v) const { return ranged_iterator<N>(num + v); }

  bool operator==(ranged_iterator<N> other) const { return num == other.num; }
  bool operator!=(ranged_iterator<N> other) const { return *this != other; }
  bool operator<(ranged_iterator<N> other) const { return num < other.num; }

  reference operator*() const { return num;}
  difference_type operator-(const ranged_iterator<N> &it) const { return num - it.num; }
  value_type operator[](const difference_type &i) const { return num + i; }
};

template <class T>
class STDIndicesStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

#if defined(ONEDPL_USE_DPCPP_BACKEND)
    // SYCL oneDPL backend
    using ExecutionPolicy = oneapi::dpl::execution::device_policy<>;
    using Allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
    using IteratorType = oneapi::dpl::counting_iterator<int>;
#elif defined(USE_ONEDPL)
    // every other non-SYCL oneDPL backend (i.e TBB, OMP)
    using ExecutionPolicy = decltype(oneapi::dpl::execution::par_unseq);
    using Allocator = std::allocator<T>;
    using IteratorType = oneapi::dpl::counting_iterator<int>;
#else
    // normal std execution policies
    using ExecutionPolicy = decltype(std::execution::par_unseq);
    using Allocator = std::allocator<T>;
    using IteratorType = ranged_iterator<int>;
#endif

    IteratorType range_start;
    IteratorType range_end;

    ExecutionPolicy exe_policy{};
    Allocator allocator;

    // Device side pointers
    std::vector<T, Allocator> a;
    std::vector<T, Allocator> b;
    std::vector<T, Allocator> c;


  public:
    STDIndicesStream(const int, int);
    ~STDIndicesStream() = default;

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

