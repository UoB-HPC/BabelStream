// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDIndicesStream.h"
#include <iostream>

template <class T>
STDIndicesStream<T>::STDIndicesStream(const int ARRAY_SIZE, int device) :
array_size{ARRAY_SIZE}, range_start(0), range_end(array_size),
#if defined(ONEDPL_USE_DPCPP_BACKEND)
  exe_policy(oneapi::dpl::execution::make_device_policy(cl::sycl::default_selector{})),
  a(sycl::malloc_shared<T>(array_size, exe_policy.queue())),
  b(sycl::malloc_shared<T>(array_size, exe_policy.queue())),
  c(sycl::malloc_shared<T>(array_size, exe_policy.queue()))
#else
  a(new T[array_size]), b(new T[array_size]), c(new T[array_size])
#endif
{
#if USE_ONEDPL
    std::cout << "Using oneDPL backend: ";
  #if defined(ONEDPL_USE_DPCPP_BACKEND)
    std::cout << "SYCL USM (device=" << exe_policy.queue().get_device().get_info<sycl::info::device::name>() << ")";
  #elif defined(ONEDPL_USE_TBB_BACKEND)
    std::cout << "TBB";
  #elif defined(ONEDPL_USE_OPENMP_BACKEND)
    std::cout << "OpenMP";
  #else
    std::cout << "Default";
  #endif
  std::cout << std::endl;
#endif
}

template <class T>
STDIndicesStream<T>::~STDIndicesStream() {
#if defined(ONEDPL_USE_DPCPP_BACKEND)
  sycl::free(a, exe_policy.queue());
  sycl::free(b, exe_policy.queue());
  sycl::free(c, exe_policy.queue());
#else
  delete[] a;
  delete[] b;
  delete[] c;
#endif
}

template <class T>
void STDIndicesStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill(exe_policy, a, a + array_size, initA);
  std::fill(exe_policy, b, b + array_size, initB);
  std::fill(exe_policy, c, c + array_size, initC);
}

template <class T>
void STDIndicesStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  // operator = is deleted because h_* vectors may have different allocator type compared to ours
  std::copy(a, a + array_size, h_a.data());
  std::copy(b, b + array_size, h_b.data());
  std::copy(c, c + array_size, h_c.data());
}


template <class T>
void STDIndicesStream<T>::copy()
{
  // c[i] = a[i]
  std::copy(exe_policy, a, a + array_size, c);
}

template <class T>
void STDIndicesStream<T>::mul()
{
  //  b[i] = scalar * c[i];
  std::transform(exe_policy, range_start, range_end, b, [&, scalar = startScalar](int i) {
    return scalar * c[i];
  });
}

template <class T>
void STDIndicesStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::transform(exe_policy, range_start, range_end, c, [&](int i) {
    return a[i] + b[i];
  });
}

template <class T>
void STDIndicesStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::transform(exe_policy, range_start, range_end, a, [&, scalar = startScalar](int i) {
    return b[i] + scalar * c[i];
  });
}

template <class T>
void STDIndicesStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
  //  Need to do in two stages with C++11 STL.
  //  1: a[i] += b[i]
  //  2: a[i] += scalar * c[i];
  std::transform(exe_policy, range_start, range_end, a, [&, scalar = startScalar](int i) {
    return a[i] + b[i] + scalar * c[i];
  });
}
   

template <class T>
T STDIndicesStream<T>::dot()
{
  // sum = 0; sum += a[i]*b[i]; return sum;
  return std::transform_reduce(exe_policy, a, a + array_size, b, 0.0);
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
template class STDIndicesStream<float>;
template class STDIndicesStream<double>;

