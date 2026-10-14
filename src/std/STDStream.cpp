// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDStream.h"
#include <algorithm>
#include <execution>

#if defined(DATA23) || defined(INDICES)
#include <ranges>
#endif

 // OneDPL workaround; TODO: remove this eventually
#include "dpl_shim.h"

#ifdef INDICES
// NVHPC workaround: TODO: remove this eventually 
#if defined(__NVCOMPILER) && defined(_NVHPC_STDPAR_GPU)
#define WORKAROUND   
#include <thrust/iterator/counting_iterator.h>
auto counting_iter(intptr_t i) { return thrust::counting_iterator<intptr_t>(i); }
auto counting_range(intptr_t b, intptr_t e) {
  struct R {
    thrust::counting_iterator<intptr_t> b, e;
    thrust::counting_iterator<intptr_t> begin() { return b; }
    thrust::counting_iterator<intptr_t> end() { return e; }
  };
  return R { .b = counting_iter(b), .e = counting_iter(e) };
}
#else  // NVHPC Workaround
auto counting_iter(intptr_t i) { return std::views::iota(i).begin(); }
auto counting_range(intptr_t b, intptr_t e) { return std::views::iota(b, e); }
#endif // NVHPC Workaround
#endif // INDICES  

template <class T>
STDStream<T>::STDStream(BenchId bs, const intptr_t array_size, const int device_id,
			      T initA, T initB, T initC)
  noexcept : array_size{array_size},
  a(alloc_raw<T>(array_size)), b(alloc_raw<T>(array_size)), c(alloc_raw<T>(array_size))
{
    std::cout << "Backing storage typeid: " << typeid(a).name() << std::endl;
#ifdef USE_ONEDPL
    std::cout << "Using oneDPL backend: ";
#if ONEDPL_USE_DPCPP_BACKEND
    std::cout << "SYCL USM (device=" << exe_policy.queue().get_device().get_info<sycl::info::device::name>() << ")";
#elif ONEDPL_USE_TBB_BACKEND
    std::cout << "TBB " TBB_VERSION_STRING;
#elif ONEDPL_USE_OPENMP_BACKEND
    std::cout << "OpenMP";
#else
    std::cout << "Default";
#endif
    std::cout << std::endl;
#endif

#ifdef WORKAROUND
    std::cout << "Non-conforming implementation: requires non-portable workarounds to run STREAM" << std::endl;
#endif      
    init_arrays(initA, initB, initC);
}

template<class T>
STDStream<T>::~STDStream() {
  dealloc_raw(a);
  dealloc_raw(b);
  dealloc_raw(c);
}

template <class T>
void STDStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill_n(exe_policy, a, array_size, initA);
  std::fill_n(exe_policy, b, array_size, initB);
  std::fill_n(exe_policy, c, array_size, initC);
}

template <class T>
void STDStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
  h_a = a;
  h_b = b;
  h_c = c;
}

template <class T>
void STDStream<T>::copy()
{
  // c[i] = a[i]
#if defined(DATA17) || defined(DATA23)
  std::copy(exe_policy, a, a + array_size, c);
#elif INDICES
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a,c=c](intptr_t i) {
      c[i] = a[i];
  });
#else
  #error unimplemented
#endif  
}

template <class T>
void STDStream<T>::mul()
{
  //  b[i] = scalar * c[i];
#if defined(DATA17) || defined(DATA23)  
  std::transform(exe_policy, c, c + array_size, b, [](T ci){ return startScalar*ci; });
#elif INDICES
  std::for_each_n(exe_policy, counting_iter(0), array_size, [b=b, c=c](intptr_t i) {
    b[i] = startScalar * c[i];
  });
#else
  #error unimplemented
#endif  
}

template <class T>
void STDStream<T>::add()
{
  //  c[i] = a[i] + b[i];
#if defined(DATA17) || defined(DATA23)  
  std::transform(exe_policy, a, a + array_size, b, c, std::plus<T>());
#elif INDICES
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a, b=b, c=c](intptr_t i) {
      c[i] = a[i] + b[i];
  });
#else
  #error unimplemented
#endif  
}

template <class T>
void STDStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
#if defined(DATA17) || defined(DATA23)
  std::transform(exe_policy, b, b + array_size, c, a, [scalar = startScalar](T bi, T ci){ return bi+scalar*ci; });
#elif INDICES
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a, b=b, c=c](intptr_t i) {
      a[i] = b[i] + startScalar * c[i];
  });
#else
  #error unimplemented
#endif  
}

template <class T>
void STDStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
#if defined(DATA17)
  //  Need to do in two round-trips with C++17 STL.
  //  1: a[i] += b[i]
  //  2: a[i] += scalar * c[i];
  std::transform(exe_policy, a, a + array_size, b, a, [](T ai, T bi){ return ai + bi; });
  std::transform(exe_policy, a, a + array_size, c, a, [](T ai, T ci){ return ai + startScalar*ci; });
#elif DATA23
  // Requires GCC 14.1 (Ubuntu 24.04):
  auto as = std::ranges::subrange(a, a + array_size);
  auto bs = std::ranges::subrange(b, b + array_size);
  auto cs = std::ranges::subrange(c, c + array_size);
  auto r = std::views::zip(as, bs, cs);
  std::transform(exe_policy, r.begin(), r.end(), a, [](auto vs) {
      auto [a, b, c] = vs;
      return a + b + startScalar * c;
  });
#elif INDICES
  std::for_each_n(exe_policy, counting_iter(0), array_size, [a=a,b=b,c=c](intptr_t i) {
    a[i] += b[i] + startScalar * c[i];
  });
#else
  #error unimplemented
#endif  
}
   

template <class T>
T STDStream<T>::dot()
{
#if defined(DATA17) || defined(DATA23)  
  // sum = 0; sum += a[i] * b[i]; return sum;
  return std::transform_reduce(exe_policy, a, a + array_size, b, T{0});
#elif INDICES
  auto r = counting_range(intptr_t(0), array_size);
  return std::transform_reduce(exe_policy, r.begin(), r.end(), T{0}, std::plus<T>{}, [a=a, b=b](intptr_t i) {
      return a[i] * b[i];
  });
#else
  #error unimplemented
#endif
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
template class STDStream<float>;
template class STDStream<double>;
