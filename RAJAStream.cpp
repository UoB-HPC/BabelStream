
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
RAJAStream<T>::RAJAStream(const int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE), range(0, ARRAY_SIZE)
{

#ifdef RAJA_TARGET_CPU
  d_a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  d_b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  d_c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
#else
  cudaMallocManaged((void**)&d_a, sizeof(T)*ARRAY_SIZE, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&d_b, sizeof(T)*ARRAY_SIZE, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&d_c, sizeof(T)*ARRAY_SIZE, cudaMemAttachGlobal);
  cudaDeviceSynchronize();
#endif
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
void RAJAStream<T>::read_arrays(
        std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  std::copy(d_a, d_a + array_size, a.data());
  std::copy(d_b, d_b + array_size, b.data());
  std::copy(d_c, d_c + array_size, c.data());
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
  // TODO implement me!
  std::cerr << "Not implemented yet!" << std::endl;
  throw std::runtime_error("Not implemented yet!");
}

template <class T>
T RAJAStream<T>::dot()
{
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;

  RAJA::ReduceSum<reduce_policy, T> sum(0.0);

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
