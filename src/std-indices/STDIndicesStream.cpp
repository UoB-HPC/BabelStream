// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDIndicesStream.h"
#include <algorithm>
#include <execution>

 // OneDPL workaround; TODO: remove this eventually
#include "dpl_shim.h"

template <class T>
STDIndicesStream<T>::STDIndicesStream(BenchId bs, const intptr_t array_size, const int device_id,
			      T initA, T initB, T initC)
  noexcept : array_size{array_size}, range(0, array_size),
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
STDIndicesStream<T>::~STDIndicesStream() {
  dealloc_raw(a);
  dealloc_raw(b);
  dealloc_raw(c);
}

template <class T>
void STDIndicesStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill_n(exe_policy, a, array_size, initA);
  std::fill_n(exe_policy, b, array_size, initB);
  std::fill_n(exe_policy, c, array_size, initC);
}

template <class T>
void STDIndicesStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
  h_a = a;
  h_b = b;
  h_c = c;
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
  std::transform(exe_policy, range.begin(), range.end(), b, [c = this->c, scalar = startScalar](intptr_t i) {
    return scalar * c[i];
  });
}

template <class T>
void STDIndicesStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::transform(exe_policy, range.begin(), range.end(), c, [a = this->a, b = this->b](intptr_t i) {
    return a[i] + b[i];
  });
}

template <class T>
void STDIndicesStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::transform(exe_policy, range.begin(), range.end(), a, [b = this->b, c = this->c, scalar = startScalar](intptr_t i) {
    return b[i] + scalar * c[i];
  });
}

template <class T>
void STDIndicesStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
  std::transform(exe_policy, range.begin(), range.end(), a, [a = this->a, b = this->b, c = this->c, scalar = startScalar](intptr_t i) {
    return a[i] + b[i] + scalar * c[i];
  });
}

template <class T>
T STDIndicesStream<T>::dot()
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
template class STDIndicesStream<float>;
template class STDIndicesStream<double>;
