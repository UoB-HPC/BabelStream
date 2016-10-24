
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "ACCStream.h"

template <class T>
ACCStream<T>::ACCStream(const unsigned int ARRAY_SIZE, T *a, T *b, T *c, int device)
{

  acc_set_device_num(device, acc_device_nvidia);

  array_size = ARRAY_SIZE;

  // Set up data region on device
  this->a = a;
  this->b = b;
  this->c = c;
  #pragma acc enter data create(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
ACCStream<T>::~ACCStream()
{
  // End data region on device
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma acc exit data delete(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
void ACCStream<T>::write_arrays(const std::vector<T>& h_a, const std::vector<T>& h_b, const std::vector<T>& h_c)
{
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma acc update device(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
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
  unsigned int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict c = this->c;
  #pragma acc kernels present(a[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void ACCStream<T>::mul()
{
  const T scalar = startScalar;

  unsigned int array_size = this->array_size;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc kernels present(b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void ACCStream<T>::add()
{
  unsigned int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc kernels present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void ACCStream<T>::triad()
{
  const T scalar = startScalar;

  unsigned int array_size = this->array_size;
  T * restrict a = this->a;
  T * restrict b = this->b;
  T * restrict c = this->c;
  #pragma acc kernels present(a[0:array_size], b[0:array_size], c[0:array_size]) wait
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}
void listDevices(void)
{
  // Get number of devices
  int count = acc_get_num_devices(acc_device_nvidia);

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
