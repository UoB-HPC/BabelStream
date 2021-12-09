
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "ACCStream.h"

template <class T>
ACCStream<T>::ACCStream(const int ARRAY_SIZE, int device)
{
  acc_device_t device_type = acc_get_device_type();
  acc_set_device_num(device, device_type);

  array_size = ARRAY_SIZE;

  // Set up data region on device
  this->a = new T[array_size];
  this->b = new T[array_size];
  this->c = new T[array_size];

  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;

  #pragma acc enter data create(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
ACCStream<T>::~ACCStream()
{
  // End data region on device
  int array_size = this->array_size;

  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;

  #pragma acc exit data delete(a[0:array_size], b[0:array_size], c[0:array_size])
  {}

  delete[] a;
  delete[] b;
  delete[] c;
}

template <class T>
void ACCStream<T>::init_arrays(T initA, T initB, T initC)
{
  int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
}

template <class T>
void ACCStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma acc update host(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
void ACCStream<T>::copy()
{
  int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void ACCStream<T>::mul()
{
  const T scalar = startScalar;

  int array_size = this->array_size;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void ACCStream<T>::add()
{
  int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void ACCStream<T>::triad()
{
  const T scalar = startScalar;

  int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void ACCStream<T>::nstream()
{
  const T scalar = startScalar;

  int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc parallel loop present(a[0:array_size],  b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
T ACCStream<T>::dot()
{
  T sum = 0.0;

  int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  #pragma acc parallel loop reduction(+:sum) present(a[0:array_size], b[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  acc_device_t device_type = acc_get_device_type();
  int count = acc_get_num_devices(device_type);

  // Print device list
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << "There are " << count << " devices." << std::endl;
  }
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class ACCStream<float>;
template class ACCStream<double>;
