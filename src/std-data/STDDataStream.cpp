// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDDataStream.h"
#include <algorithm>
#include <execution>

 // OneDPL workaround; TODO: remove this eventually
#include "dpl_shim.h"

template <class T>
STDDataStream<T>::STDDataStream(BenchId bs, const intptr_t array_size, const int device_id,
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
STDDataStream<T>::~STDDataStream() {
  dealloc_raw(a);
  dealloc_raw(b);
  dealloc_raw(c);
}

template <class T>
void STDDataStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill_n(exe_policy, a, array_size, initA);
  std::fill_n(exe_policy, b, array_size, initB);
  std::fill_n(exe_policy, c, array_size, initC);
}

template <class T>
void STDDataStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
  h_a = a;
  h_b = b;
  h_c = c;
}

template <class T>
void STDDataStream<T>::copy()
{
  // c[i] = a[i]
  std::copy(exe_policy, a, a + array_size, c);
}

template <class T>
void STDDataStream<T>::mul()
{
  //  b[i] = scalar * c[i];
  std::transform(exe_policy, c, c + array_size, b, [](T ci){ return startScalar*ci; });
}

template <class T>
void STDDataStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::transform(exe_policy, a, a + array_size, b, c, std::plus<T>());
}

template <class T>
void STDDataStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::transform(exe_policy, b, b + array_size, c, a, [scalar = startScalar](T bi, T ci){ return bi+scalar*ci; });
}

template <class T>
void STDDataStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
  //  Need to do in two round-trips with C++17 STL.
  //  1: a[i] += b[i]
  //  2: a[i] += scalar * c[i];
  std::transform(exe_policy, a, a + array_size, b, a, [](T ai, T bi){ return ai + bi; });
  std::transform(exe_policy, a, a + array_size, c, a, [](T ai, T ci){ return ai + startScalar*ci; });
}

template <class T>
T STDDataStream<T>::dot()
{
  // sum = 0; sum += a[i] * b[i]; return sum;
  return std::transform_reduce(exe_policy, a, a + array_size, b, T{0});
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
