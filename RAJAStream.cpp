
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
  d_a = new T[ARRAY_SIZE];
  d_b = new T[ARRAY_SIZE];
  d_c = new T[ARRAY_SIZE];
}

template <class T>
RAJAStream<T>::~RAJAStream()
{
  delete[] d_a;
  delete[] d_b;
  delete[] d_c;
}

template <class T>
void RAJAStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  std::copy(a.begin(), a.end(), d_a);
  std::copy(b.begin(), b.end(), d_b);
  std::copy(c.begin(), c.end(), d_c);
}

template <class T>
void RAJAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  std::copy(d_a, d_a + array_size - 1, a.data());
  std::copy(d_b, d_b + array_size - 1, b.data());
  std::copy(d_c, d_c + array_size - 1, c.data());
}

template <class T>
void RAJAStream<T>::copy()
{
  forall<policy>(index_set, [=] RAJA_DEVICE (int index) 
  {
    d_c[index] = d_a[index];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
  const T scalar = 3.0;
  forall<policy>(index_set, [=] RAJA_DEVICE (int index) 
  {
    d_b[index] = scalar*d_c[index];
  });
}

template <class T>
void RAJAStream<T>::add()
{
  forall<policy>(index_set, [=] RAJA_DEVICE (int index) 
  {
    d_c[index] = d_a[index] + d_b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
  const T scalar = 3.0;
  forall<policy>(index_set, [=] RAJA_DEVICE (int index) 
  {
    d_a[index] = d_b[index] + scalar*d_c[index];
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

