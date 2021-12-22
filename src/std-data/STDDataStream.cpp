// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDDataStream.h"

#include <algorithm>
#include <execution>
#include <numeric>

// There are three execution policies:
// auto exe_policy = std::execution::seq;
// auto exe_policy = std::execution::par;
auto exe_policy = std::execution::par_unseq;


template <class T>
STDDataStream<T>::STDDataStream(const int ARRAY_SIZE, int device)
  noexcept : array_size{ARRAY_SIZE}, a(array_size), b(array_size), c(array_size)
{
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
  h_a = a;
  h_b = b;
  h_c = c;
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

