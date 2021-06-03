// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "TBBStream.hpp"

template <class T>
TBBStream<T>::TBBStream(const int ARRAY_SIZE, int device)
 : partitioner(), range(0, ARRAY_SIZE), a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE)
{
  if(device != 0){
    throw std::runtime_error("Device != 0 is not supported by TBB");
  }
  std::cout << "Using TBB partitioner: " PARTITIONER_NAME << std::endl;
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
void TBBStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  // Element-wise copy.
  h_a = a;
  h_b = b;
  h_c = c;
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

