
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "RAJAStream.hpp"

using RAJA::forall;
using RAJA::RangeSegment;

template <class T>
RAJAStream<T>::RAJAStream(const unsigned int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE)
{
  RangeSegment seg(0, ARRAY_SIZE);
  index_set.push_back(seg);

#ifdef RAJA_TARGET_CPU
  d_a = new T[ARRAY_SIZE];
  d_b = new T[ARRAY_SIZE];
  d_c = new T[ARRAY_SIZE];
  forall<policy>(index_set, [=] RAJA_DEVICE (int index)
  {
    d_a[index] = 0.0;
    d_b[index] = 0.0;
    d_c[index] = 0.0;
  });
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
  delete[] d_a;
  delete[] d_b;
  delete[] d_c;
#else
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
#endif
}

template <class T>
void RAJAStream<T>::write_arrays(
        const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  std::copy(a.begin(), a.end(), d_a);
  std::copy(b.begin(), b.end(), d_b);
  std::copy(c.begin(), c.end(), d_c);
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
  T* a = d_a;
  T* c = d_c;
  forall<policy>(index_set, [=] RAJA_DEVICE (int index)
  {
    c[index] = a[index];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
  T* b = d_b;
  T* c = d_c;
  const T scalar = startScalar;
  forall<policy>(index_set, [=] RAJA_DEVICE (int index)
  {
    b[index] = scalar*c[index];
  });
}

template <class T>
void RAJAStream<T>::add()
{
  T* a = d_a;
  T* b = d_b;
  T* c = d_c;
  forall<policy>(index_set, [=] RAJA_DEVICE (int index)
  {
    c[index] = a[index] + b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
  T* a = d_a;
  T* b = d_b;
  T* c = d_c;
  const T scalar = startScalar;
  forall<policy>(index_set, [=] RAJA_DEVICE (int index)
  {
    a[index] = b[index] + scalar*c[index];
  });
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
