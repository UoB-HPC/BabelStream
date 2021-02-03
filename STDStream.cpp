// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDStream.h"

#include <algorithm>
#include <execution>
#include <numeric>

// There are three execution policies:
// auto exe_policy = std::execution::seq;
// auto exe_policy = std::execution::par;
auto exe_policy = std::execution::par_unseq;

template <class T>
STDStream<T>::STDStream(const int ARRAY_SIZE, int device)
  noexcept : array_size{ARRAY_SIZE}, a{new T[array_size]}, b{new T[array_size]}, c{new T[array_size]}
{
}

template <class T>
STDStream<T>::~STDStream()
{
  delete[] a;
  delete[] b;
  delete[] c;
}

template <class T>
void STDStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill(exe_policy, a, a+array_size, initA);
  std::fill(exe_policy, b, b+array_size, initB);
  std::fill(exe_policy, c, c+array_size, initC);
}

template <class T>
void STDStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  std::copy(exe_policy, a, a+array_size, h_a.data());
  std::copy(exe_policy, b, b+array_size, h_b.data());
  std::copy(exe_policy, c, c+array_size, h_c.data());
}

template <class T>
void STDStream<T>::copy()
{
  // c[i] = a[i]
  std::copy(exe_policy, a, a+array_size, c) ;
}

template <class T>
void STDStream<T>::mul()
{
  //  b[i] = scalar * c[i];
  std::transform(exe_policy, c, c+array_size, b, [](T ci){ return startScalar*ci; });
}

template <class T>
void STDStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::transform(exe_policy, a, a+array_size, b, c, std::plus<T>());
}

template <class T>
void STDStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::transform(exe_policy, b, b+array_size, c, a, [](T bi, T ci){ return bi+startScalar*ci; });
}

template <class T>
void STDStream<T>::nstream()
{
  //  a[i] += b[i] + scalar * c[i];
  //  Need to do in two stages with C++11 STL.
  //  1: a[i] += b[i]
  //  2: a[i] += scalar * c[i];
  std::transform(exe_policy, a, a+array_size, b, a, [](T ai, T bi){ return ai + bi; });
  std::transform(exe_policy, a, a+array_size, c, a, [](T ai, T ci){ return ai + startScalar*ci; });
}

template <class T>
T STDStream<T>::dot()
{
  // sum = 0; sum += a[i]*b[i]; return sum;
  return std::transform_reduce(exe_policy, a, a+array_size, b, 0.0);
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

