
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith, Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include "SerialStream.h"

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
SerialStream<T>::SerialStream(const intptr_t ARRAY_SIZE, int device)
{
  array_size = ARRAY_SIZE;

  // Allocate on the host
  this->a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
}

template <class T>
SerialStream<T>::~SerialStream()
{
  free(a);
  free(b);
  free(c);
}

template <class T>
void SerialStream<T>::init_arrays(T initA, T initB, T initC)
{
  intptr_t array_size = this->array_size;
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
}

template <class T>
void SerialStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  for (intptr_t i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
  }

}

template <class T>
void SerialStream<T>::copy()
{
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void SerialStream<T>::mul()
{
  const T scalar = startScalar;
  for (intptr_t i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void SerialStream<T>::add()
{
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void SerialStream<T>::triad()
{
  const T scalar = startScalar;
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void SerialStream<T>::nstream()
{
  const T scalar = startScalar;
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
T SerialStream<T>::dot()
{
  T sum{};
  for (intptr_t i = 0; i < array_size; i++)
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
template class SerialStream<float>;
template class SerialStream<double>;
