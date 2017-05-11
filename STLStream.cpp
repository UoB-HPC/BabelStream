
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STLStream.hpp"

#include "pstl/algorithm"
#include "pstl/execution"
#include <functional>

template <class T>
STLStream<T>::STLStream(const unsigned int ARRAY_SIZE, int device) :
  array_size(ARRAY_SIZE), a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE)
{
}

template <class T>
STLStream<T>::~STLStream()
{
}

template <class T>
void STLStream<T>::init_arrays(T initA, T initB, T initC)
{
  for (int i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
}

template <class T>
void STLStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  for (int i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
  }
}

template <class T>
void STLStream<T>::copy()
{
  std::copy(std::execution::par_unseq,
    a.begin(), a.end(), c.begin()
    );
}

template <class T>
void STLStream<T>::mul()
{
  const T scalar = startScalar;
  std::transform(std::execution::par_unseq,
    c.begin(), c.end(), b.begin(),
    [scalar](T c){
      return scalar * c;
    });
}

template <class T>
void STLStream<T>::add()
{
  std::transform(std::execution::par_unseq,
    a.begin(), a.end(), b.begin(), c.begin(),
    [](T a, T b){
      return a + b;
    });
}

template <class T>
void STLStream<T>::triad()
{
  const T scalar = startScalar;
  std::transform(std::execution::par_unseq,
    b.begin(), b.end(), c.begin(), a.begin(),
    [scalar](T b, T c) {
      return b + scalar * c;
    });
}

template <class T>
T STLStream<T>::dot()
{
  T sum = 0.0;
  for (int i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }

  return sum;
}



void listDevices(void)
{
  std::cout << "0: CPU" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}

template class STLStream<float>;
template class STLStream<double>;
