
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OMP3Stream.h"

template <class T>
OMP3Stream<T>::OMP3Stream(const unsigned int ARRAY_SIZE, T *a, T *b, T *c)
{
  array_size = ARRAY_SIZE;
  this->a = (T*)malloc(sizeof(T)*array_size);
  this->b = (T*)malloc(sizeof(T)*array_size);
  this->c = (T*)malloc(sizeof(T)*array_size);
}

template <class T>
OMP3Stream<T>::~OMP3Stream()
{
  free(a);
  free(b);
  free(c);
}


template <class T>
void OMP3Stream<T>::write_arrays(const std::vector<T>& h_a, const std::vector<T>& h_b, const std::vector<T>& h_c)
{
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    a[i] = h_a[i];
    b[i] = h_b[i];
    c[i] = h_c[i];
  }
}

template <class T>
void OMP3Stream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
  }
}

template <class T>
void OMP3Stream<T>::copy()
{
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void OMP3Stream<T>::mul()
{
  const T scalar = startScalar;
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void OMP3Stream<T>::add()
{
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void OMP3Stream<T>::triad()
{
  const T scalar = startScalar;
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
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


template class OMP3Stream<float>;
template class OMP3Stream<double>;
