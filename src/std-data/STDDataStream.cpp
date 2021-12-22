// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDDataStream.h"
#include <iostream>

template <class T>
STDDataStream<T>::STDDataStream(const int ARRAY_SIZE, int device) : array_size{ARRAY_SIZE},
#if defined(ONEDPL_USE_DPCPP_BACKEND)
    exe_policy(oneapi::dpl::execution::make_device_policy(cl::sycl::default_selector{})),
    allocator(exe_policy.queue()),
    a(array_size, allocator), b(array_size, allocator), c(array_size, allocator)
#else
    a(array_size), b(array_size),c(array_size)
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
void STDDataStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill(exe_policy, a.begin(), a.end(), initA);
  std::fill(exe_policy, b.begin(), b.end(), initB);
  std::fill(exe_policy, c.begin(), c.end(), initC);
}

template <class T>
void STDDataStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  // operator = is deleted because h_* vectors may have different allocator type compared to ours
  std::copy(a.begin(), a.end(), h_a.begin());
  std::copy(b.begin(), b.end(), h_b.begin());
  std::copy(c.begin(), c.end(), h_c.begin());
}

template <class T>
void STDDataStream<T>::copy()
{
  // c[i] = a[i]
  std::copy(exe_policy, a.begin(), a.end(), c.begin());
}

template <class T>
void STDDataStream<T>::mul()
{
  //  b[i] = scalar * c[i];
  std::transform(exe_policy, c.begin(), c.end(), b.begin(), [scalar = startScalar](T ci){ return scalar*ci; });
}

template <class T>
void STDDataStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::transform(exe_policy, a.begin(), a.end(), b.begin(), c.begin(), std::plus<T>());
}

template <class T>
void STDDataStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::transform(exe_policy, b.begin(), b.end(), c.begin(), a.begin(), [scalar = startScalar](T bi, T ci){ return bi+scalar*ci; });
}

template <class T>
void STDDataStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
  //  Need to do in two stages with C++11 STL.
  //  1: a[i] += b[i]
  //  2: a[i] += scalar * c[i];
  std::transform(exe_policy, a.begin(), a.end(), b.begin(), a.begin(), [](T ai, T bi){ return ai + bi; });
  std::transform(exe_policy, a.begin(), a.end(), c.begin(), a.begin(), [scalar = startScalar](T ai, T ci){ return ai + scalar*ci; });
}
   

template <class T>
T STDDataStream<T>::dot()
{
  // sum = 0; sum += a[i]*b[i]; return sum;
  return std::transform_reduce(exe_policy, a.begin(), a.end(), b.begin(), 0.0);
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
template class STDDataStream<float>;
template class STDDataStream<double>;

