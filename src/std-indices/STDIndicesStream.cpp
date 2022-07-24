// Copyright (c) 2021 Tom Deakin and Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STDIndicesStream.h"

#include <algorithm>
#include <execution>
#include <numeric>

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

#ifdef USE_VECTOR
#define BEGIN(x) (x).begin()
#define END(x) (x).end()
#else
#define BEGIN(x) (x)
#define END(x) ((x) + array_size)
#endif

// There are three execution policies:
// auto exe_policy = std::execution::seq;
// auto exe_policy = std::execution::par;
constexpr auto exe_policy = std::execution::par_unseq;

template <class T>
STDIndicesStream<T>::STDIndicesStream(const int ARRAY_SIZE, int device)
  noexcept : array_size{ARRAY_SIZE}, range(0, array_size),
#ifdef USE_VECTOR
  a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE)
#else
  a((T *) aligned_alloc(ALIGNMENT, sizeof(T) * ARRAY_SIZE)),
  b((T *) aligned_alloc(ALIGNMENT, sizeof(T) * ARRAY_SIZE)),
  c((T *) aligned_alloc(ALIGNMENT, sizeof(T) * ARRAY_SIZE))
#endif
{ std::cout <<"Backing storage typeid: " << typeid(a).name() << std::endl; }

template <class T>
void STDIndicesStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::fill(exe_policy, BEGIN(a), END(a), initA);
  std::fill(exe_policy, BEGIN(b), END(b), initB);
  std::fill(exe_policy, BEGIN(c), END(c), initC);
}

template <class T>
void STDIndicesStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  std::copy(BEGIN(a), END(a), h_a.begin());
  std::copy(BEGIN(b), END(b), h_b.begin());
  std::copy(BEGIN(c), END(c), h_c.begin());
}

template <class T>
void STDIndicesStream<T>::copy()
{
  // c[i] = a[i]
  std::copy(exe_policy, BEGIN(a), END(a), BEGIN(c));
}

template <class T>
void STDIndicesStream<T>::mul()
{
  //  b[i] = scalar * c[i];
  std::transform(exe_policy, range.begin(), range.end(), BEGIN(b), [&, scalar = startScalar](int i) {
    return scalar * c[i];
  });
}

template <class T>
void STDIndicesStream<T>::add()
{
  //  c[i] = a[i] + b[i];
  std::transform(exe_policy, range.begin(), range.end(), BEGIN(c), [&](int i) {
    return a[i] + b[i];
  });
}

template <class T>
void STDIndicesStream<T>::triad()
{
  //  a[i] = b[i] + scalar * c[i];
  std::transform(exe_policy, range.begin(), range.end(), BEGIN(a), [&, scalar = startScalar](int i) {
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
  std::transform(exe_policy, range.begin(), range.end(), BEGIN(a), [&, scalar = startScalar](int i) {
    return a[i] + b[i] + scalar * c[i];
  });
}
   

template <class T>
T STDIndicesStream<T>::dot()
{
  // sum = 0; sum += a[i]*b[i]; return sum;
  return std::transform_reduce(exe_policy, BEGIN(a), END(a), BEGIN(b), 0.0);
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

#undef BEGIN
#undef END
