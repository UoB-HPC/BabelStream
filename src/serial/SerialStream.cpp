
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith, Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SerialStream.h"
#include "Alloc.h"

template <class T>
SerialStream<T>::SerialStream(BenchId bs, const intptr_t array_size, const int device_id,
			      T initA, T initB, T initC)
  : array_size{array_size}
{
  // Allocate on the host
  this->a = alloc_raw<T>(array_size);
  this->b = alloc_raw<T>(array_size);
  this->c = alloc_raw<T>(array_size);

  init_arrays(initA, initB, initC);
}

template <class T>
SerialStream<T>::~SerialStream()
{
  dealloc_raw(a);
  dealloc_raw(b);
  dealloc_raw(c);
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
void SerialStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
  h_a = a;
  h_b = b;
  h_c = c;
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
