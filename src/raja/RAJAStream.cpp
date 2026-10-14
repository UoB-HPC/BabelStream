
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include <stdexcept>
#include "RAJAStream.hpp"

using RAJA::forall;

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
RAJAStream<T>::RAJAStream(BenchId bs, const intptr_t array_size, const int device_index,
			  T initA, T initB, T initC)
  : array_size(array_size), range(0, array_size)
{

#ifdef RAJA_TARGET_CPU
  d_a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  d_b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  d_c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
#else
  cudaMallocManaged((void**)&d_a, sizeof(T)*array_size, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&d_b, sizeof(T)*array_size, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&d_c, sizeof(T)*array_size, cudaMemAttachGlobal);
  cudaDeviceSynchronize();
#endif

  init_arrays(initA, initB, initC);
}

template <class T>
RAJAStream<T>::~RAJAStream()
{
#ifdef RAJA_TARGET_CPU
  free(d_a);
  free(d_b);
  free(d_c);
#else
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
#endif
}

template <class T>
void RAJAStream<T>::init_arrays(T initA, T initB, T initC)
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    a[index] = initA;
    b[index] = initB;
    c[index] = initC;
  });
}

template <class T>
void RAJAStream<T>::get_arrays(T const*& a, T const*& b, T const*& c)
{
  a = d_a;
  b = d_b;
  c = d_c;
}

template <class T>
void RAJAStream<T>::copy()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT c = d_c;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    b[index] = scalar*c[index];
  });
}

template <class T>
void RAJAStream<T>::add()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index] + b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    a[index] = b[index] + scalar*c[index];
  });
}

template <class T>
void RAJAStream<T>::nstream()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    a[index] += b[index] + scalar * c[index];;
  });
}

template <class T>
T RAJAStream<T>::dot()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;

  RAJA::ReduceSum<reduce_policy, T> sum(T{});

  forall<policy>(range, [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    sum += a[index] * b[index];
  });

  return T(sum);
}


void listDevices(void)
{
  std::cout << "This is not the device you are looking for.";
}


std::string getDeviceName(const int device)
{
  return "RAJA";
}


std::string getDeviceDriver(const int device)
{
  return "RAJA";
}

template class RAJAStream<float>;
template class RAJAStream<double>;
