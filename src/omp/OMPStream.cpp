
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include "OMPStream.h"

#if defined(PAGEFAULT)
#pragma omp requires unified_shared_memory
#endif

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
OMPStream<T>::OMPStream(BenchId bs, const intptr_t array_size, const int device,
			T initA, T initB, T initC)
  : array_size(array_size)
{
  // Allocate on the host
  this->a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);

#ifdef OMP_TARGET_GPU
  omp_set_default_device(device);
  #if !defined(PAGEFAULT)
    T *a = this->a;
    T *b = this->b;
    T *c = this->c;
    // Set up data region on device
    #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size])
    {}
  #endif
#endif

  init_arrays(initA, initB, initC);
}

template <class T>
OMPStream<T>::~OMPStream()
{
#if defined(OMP_TARGET_GPU) && !defined(PAGEFAULT)
  // End data region on device
  intptr_t array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#endif
  free(a);
  free(b);
  free(c);
}

template <class T>
void OMPStream<T>::init_arrays(T initA, T initB, T initC)
{
  intptr_t array_size = this->array_size;
#if defined(OMP_TARGET_GPU) && !defined(PAGEFAULT)
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC) && !defined(PAGEFAULT)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{

#if defined(OMP_TARGET_GPU) && !defined(PAGEFAULT)
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
  {}
#endif
  h_a = a;
  h_b = b;
  h_c = c;
}

template <class T>
void OMPStream<T>::copy()
{
#if defined(OMP_TARGET_GPU) && !defined(PAGEFAULT)
  intptr_t array_size = this->array_size;
  T *a = this->a;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC) && !defined(PAGEFAULT)
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
  #if !defined(PAGEFAULT)
    intptr_t array_size = this->array_size;
    T *b = this->b;
    T *c = this->c;
  #endif
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (intptr_t i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC) && !defined(PAGEFAULT)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(c[0:0])
  #endif
}

template <class T>
void OMPStream<T>::add()
{
#ifdef OMP_TARGET_GPU
  #if !defined(PAGEFAULT)
    intptr_t array_size = this->array_size;
    T *a = this->a;
    T *b = this->b;
    T *c = this->c;
  #endif
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (intptr_t i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC) && !defined(PAGEFAULT)
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
  #if !defined(PAGEFAULT)
    intptr_t array_size = this->array_size;
    T *a = this->a;
    T *b = this->b;
    T *c = this->c;
  #endif
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC) && !defined(PAGEFAULT)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
void OMPStream<T>::nstream()
{
  const T scalar = startScalar;

#ifdef OMP_TARGET_GPU
  #if !defined(PAGEFAULT)
    intptr_t array_size = this->array_size;
    T *a = this->a;
    T *b = this->b;
    T *c = this->c;
  #endif
  #pragma omp target teams distribute parallel for simd
#else
  #pragma omp parallel for
#endif
  for (intptr_t i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
  #if defined(OMP_TARGET_GPU) && defined(_CRAYC) && !defined(PAGEFAULT)
  // If using the Cray compiler, the kernels do not block, so this update forces
  // a small copy to ensure blocking so that timing is correct
  #pragma omp target update from(a[0:0])
  #endif
}

template <class T>
T OMPStream<T>::dot()
{
  T sum{};

#ifdef OMP_TARGET_GPU
  #if !defined(PAGEFAULT)
    intptr_t array_size = this->array_size;
    T *a = this->a;
    T *b = this->b;
  #endif
  #pragma omp target teams distribute parallel for simd map(tofrom: sum) reduction(+:sum)
#else
  #pragma omp parallel for reduction(+:sum)
#endif
  for (intptr_t i = 0; i < array_size; i++)
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
