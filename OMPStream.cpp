
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OMPStream.h"

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
OMPStream<T>::OMPStream(const unsigned int ARRAY_SIZE, T *a, T *b, T *c, int device)
{
  array_size = ARRAY_SIZE;

#ifdef OMP_TARGET_GPU
  omp_set_default_device(device);
  // Set up data region on device
  this->a = a;
  this->b = b;
  this->c = c;
  #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#else
  // Allocate on the host
  this->a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
#endif
}

template <class T>
OMPStream<T>::~OMPStream()
{
#ifdef OMP_TARGET_GPU
  // End data region on device
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#else
  free(a);
  free(b);
  free(c);
#endif
}

template <class T>
void OMPStream<T>::init_arrays(T initA, T initB, T initC)
{
  unsigned int array_size = this->array_size;
#ifdef OMP_TARGET_GPU
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (int i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
#ifdef OMP_TARGET_GPU
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#else
  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
  }
#endif
}

template <class T>
void OMPStream<T>::copy()
{
#ifdef OMP_TARGET_GPU
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::mul()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  unsigned int array_size = this->array_size;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(c[0:0])
  #endif
}

template <class T>
void OMPStream<T>::add()
{
#ifdef OMP_TARGET_GPU
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::triad()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
T OMPStream<T>::dot()
{
  T sum = 0.0;

#ifdef OMP_TARGET_GPU
  unsigned int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  #pragma omp target teams distribute parallel for simd map(tofrom: sum) reduction(+:sum)
#else
  #pragma omp parallel for reduction(+:sum)
#endif
  for (int i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }

  return sum;
}



void listDevices(void)
{
#ifdef OMP_TARGET_GPU
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
#else
  std::cout << "0: CPU" << std::endl;
#endif
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class OMPStream<float>;
template class OMPStream<double>;
