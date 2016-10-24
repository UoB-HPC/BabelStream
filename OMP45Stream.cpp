
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OMP45Stream.h"

template <class T>
OMP45Stream<T>::OMP45Stream(const unsigned int ARRAY_SIZE, T *a, T *b, T *c, int device)
{
  omp_set_default_device(device);

  array_size = ARRAY_SIZE;

  // Set up data region on device
  this->a = a;
  this->b = b;
  this->c = c;
  #pragma omp target enter data map(to: a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
OMP45Stream<T>::~OMP45Stream()
{
  // End data region on device
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
void OMP45Stream<T>::write_arrays(const std::vector<T>& h_a, const std::vector<T>& h_b, const std::vector<T>& h_c)
{
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target update to(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
void OMP45Stream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
}

template <class T>
void OMP45Stream<T>::copy()
{
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd map(to: a[0:array_size], c[0:array_size])
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void OMP45Stream<T>::mul()
{
  const T scalar = startScalar;

  unsigned int array_size = this->array_size;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd map(to: b[0:array_size], c[0:array_size])
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void OMP45Stream<T>::add()
{
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd map(to: a[0:array_size], b[0:array_size], c[0:array_size])
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void OMP45Stream<T>::triad()
{
  const T scalar = startScalar;

  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd map(to: a[0:array_size], b[0:array_size], c[0:array_size])
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}
void listDevices(void)
{
  // Get number of devices
  int count = omp_get_num_devices();

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
template class OMP45Stream<float>;
template class OMP45Stream<double>;
