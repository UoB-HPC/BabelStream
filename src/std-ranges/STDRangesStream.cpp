// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDRangesStream.h"
#include <algorithm>
#include <execution>
#include <ranges>

 // OneDPL workaround; TODO: remove this eventually
#include "dpl_shim.h"

// views::iota iterators are not usable in device code for NVHPC, hipstdpar, and the oneDPL SYCL backend
#if (defined(__NVCOMPILER) && defined(_NVHPC_STDPAR_GPU)) || defined(__HIPSTDPAR__)
#define WORKAROUND
#define COUNTING_ITERATOR thrust::counting_iterator
#include <thrust/iterator/counting_iterator.h>
#elif defined(USE_ONEDPL) && ONEDPL_USE_DPCPP_BACKEND
#define COUNTING_ITERATOR oneapi::dpl::counting_iterator
#include <oneapi/dpl/iterator>
#endif

#ifdef COUNTING_ITERATOR
auto counting_iter(intptr_t i) { return COUNTING_ITERATOR<intptr_t>(i); }
auto counting_range(intptr_t b, intptr_t e) {
  struct R {
    COUNTING_ITERATOR<intptr_t> b, e;
    COUNTING_ITERATOR<intptr_t> begin() { return b; }
    COUNTING_ITERATOR<intptr_t> end() { return e; }
  };
  return R { .b = counting_iter(b), .e = counting_iter(e) };
}
#else
auto counting_iter(intptr_t i) { return std::views::iota(i).begin(); }
auto counting_range(intptr_t b, intptr_t e) { return std::views::iota(b, e); }
#endif

template <class T>
STDRangesStream<T>::STDRangesStream(BenchId bs, const intptr_t array_size, const int device_id,
			      T initA, T initB, T initC)
  noexcept : array_size{array_size},
  a(alloc_raw<T>(array_size)), b(alloc_raw<T>(array_size)), c(alloc_raw<T>(array_size))
{
    std::cout << "Backing storage typeid: " << typeid(a).name() << std::endl;
#ifdef USE_ONEDPL
    print_dpl_backend();
#endif

#ifdef WORKAROUND
    std::cout << "Non-conforming implementation: requires non-portable workarounds to run STREAM" << std::endl;
#endif      
    init_arrays(initA, initB, initC);
}

template<class T>
STDRangesStream<T>::~STDRangesStream() {
  dealloc_raw(a);
  dealloc_raw(b);
  dealloc_raw(c);
}

template <class T>
void STDRangesStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill_n(exe_policy, a, array_size, initA);
  std::fill_n(exe_policy, b, array_size, initB);
  std::fill_n(exe_policy, c, array_size, initC);
}

template <class T>
void STDRangesStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
  h_a = a;
  h_b = b;
  h_c = c;
}

template <class T>
void STDRangesStream<T>::copy()
{
  // c[i] = a[i]
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a,c=c](intptr_t i) {
      c[i] = a[i];
  });
}

template <class T>
void STDRangesStream<T>::mul()
{
  //  b[i] = scalar * c[i];
  std::for_each_n(exe_policy, counting_iter(0), array_size, [b=b, c=c](intptr_t i) {
    b[i] = startScalar * c[i];
  });
}

template <class T>
void STDRangesStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a, b=b, c=c](intptr_t i) {
      c[i] = a[i] + b[i];
  });
}

template <class T>
void STDRangesStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a, b=b, c=c](intptr_t i) {
      a[i] = b[i] + startScalar * c[i];
  });
}

template <class T>
void STDRangesStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a,b=b,c=c](intptr_t i) {
    a[i] += b[i] + startScalar * c[i];
  });
}

template <class T>
T STDRangesStream<T>::dot()
{
  auto r = counting_range(intptr_t(0), array_size);
  return std::transform_reduce(exe_policy, r.begin(), r.end(), T{0}, std::plus<T>{}, [a=a, b=b](intptr_t i) {
      return a[i] * b[i];
  });
}

void listDevices(void)
{
  std::cout << "Listing devices is not supported by the Parallel STL" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class STDRangesStream<float>;
template class STDRangesStream<double>;
