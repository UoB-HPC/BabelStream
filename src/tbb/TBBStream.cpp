// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "TBBStream.hpp"
#include <cstdlib>

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

#ifdef USE_VECTOR
#define BEGIN(x) (x).begin()
#define END(x) (x).end()
#else
#define BEGIN(x) (x)
#define END(x) ((x) + array_size)
#endif

template <class T>
TBBStream<T>::TBBStream(BenchId bs, const intptr_t array_size, const int device,
			T initA, T initB, T initC)
  : partitioner(), range(0, (size_t)array_size),
#ifdef USE_VECTOR
   a(array_size), b(array_size), c(array_size)
#else
   array_size(array_size),
   a((T *) aligned_alloc(ALIGNMENT, sizeof(T) * array_size)),
   b((T *) aligned_alloc(ALIGNMENT, sizeof(T) * array_size)),
   c((T *) aligned_alloc(ALIGNMENT, sizeof(T) * array_size))
#endif
{
  if(device != 0){
    throw std::runtime_error("Device != 0 is not supported by TBB");
  }
  std::cout << "Using TBB partitioner: " PARTITIONER_NAME << std::endl;
  std::cout << "Backing storage typeid: " << typeid(a).name() << std::endl;

  init_arrays(initA, initB, initC);
}


template <class T>
void TBBStream<T>::init_arrays(T initA, T initB, T initC)
{

  tbb::parallel_for(range, [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      a[i] = initA;
      b[i] = initB;
      c[i] = initC;
    }
  }, partitioner);

}

template <class T>
void TBBStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
#ifdef USE_VECTOR
  h_a = a.data();
  h_b = b.data();
  h_c = c.data();
#else
  h_a = a;
  h_b = b;
  h_c = c;
#endif
}

template <class T>
void TBBStream<T>::copy()
{
  tbb::parallel_for(range, [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
       c[i] = a[i];
    }
  }, partitioner);
}

template <class T>
void TBBStream<T>::mul()
{
  const T scalar = startScalar;

  tbb::parallel_for(range, [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
       b[i] = scalar * c[i];
    }
  }, partitioner);

}

template <class T>
void TBBStream<T>::add()
{

  tbb::parallel_for(range, [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
       c[i] = a[i] + b[i];
    }
  }, partitioner);

}

template <class T>
void TBBStream<T>::triad()
{
  const T scalar = startScalar;

  tbb::parallel_for(range, [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
       a[i] = b[i] + scalar * c[i];
    }
  }, partitioner);

}

template <class T>
void TBBStream<T>::nstream()
{
  const T scalar = startScalar;

  tbb::parallel_for(range, [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
       a[i] += b[i] + scalar * c[i];
    }
  }, partitioner);

}

template <class T>
T TBBStream<T>::dot()
{
  // sum += a[i] * b[i];
  return
    tbb::parallel_reduce(range, T{}, [&](const tbb::blocked_range<size_t>& r, T acc) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        acc += a[i] * b[i];
      }
      return acc;
    }, std::plus<T>(), partitioner);
}

void listDevices(void)
{
   std::cout << "Listing devices is not supported by TBB" << std::endl;
}

std::string getDeviceName(const int device)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}

template class TBBStream<float>;
template class TBBStream<double>;

#undef BEGIN
#undef END
