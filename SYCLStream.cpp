
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream.h"

#include <iostream>

using namespace cl::sycl;

template <class T>
SYCLStream<T>::SYCLStream(const unsigned int ARRAY_SIZE, const int device_index)
{
  array_size = ARRAY_SIZE;

  // Create buffers
  d_a = buffer<T>(array_size);
  d_b = buffer<T>(array_size);
  d_c = buffer<T>(array_size);
}

template <class T>
SYCLStream<T>::~SYCLStream()
{
}

template <class T>
void SYCLStream<T>::copy()
{
  queue.submit([&](handler &cgh)
  {
    auto ka = d_a.template get_access<access::read>(cgh);
    auto kc = d_c.template get_access<access::write>(cgh);
    cgh.parallel_for(range<1>{array_size}, [=](id<1> index)
    {
      kc[index] = ka[index];
    });
  });
  queue.wait();
}

template <class T>
void SYCLStream<T>::mul()
{
  const T scalar = 3.0;
  queue.submit([&](handler &cgh)
  {
    auto kb = d_b.template get_access<access::write>(cgh);
    auto kc = d_c.template get_access<access::read>(cgh);
    cgh.parallel_for(range<1>{array_size}, [=](id<1> index)
    {
      kb[index] = scalar * kc[index];
    });
  });
  queue.wait();
}

template <class T>
void SYCLStream<T>::add()
{
  queue.submit([&](handler &cgh)
  {
    auto ka = d_a.template get_access<access::read>(cgh);
    auto kb = d_b.template get_access<access::read>(cgh);
    auto kc = d_c.template get_access<access::write>(cgh);
    cgh.parallel_for(range<1>{array_size}, [=](id<1> index)
    {
      kc[index] = ka[index] + kb[index];
    });
  });
  queue.wait();
}

template <class T>
void SYCLStream<T>::triad()
{
  const T scalar = 3.0;
  queue.submit([&](handler &cgh)
  {
    auto ka = d_a.template get_access<access::write>(cgh);
    auto kb = d_b.template get_access<access::read>(cgh);
    auto kc = d_c.template get_access<access::read>(cgh);
    cgh.parallel_for(range<1>{array_size}, [=](id<1> index){
      ka[index] = kb[index] + scalar * kc[index];
    });
  });
  queue.wait();
}

template <class T>
void SYCLStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  auto _a = d_a.template get_access<access::write>();
  auto _b = d_b.template get_access<access::write>();
  auto _c = d_c.template get_access<access::write>();
  for (int i = 0; i < array_size; i++)
  {
    _a[i] = a[i];
    _b[i] = b[i];
    _c[i] = c[i];
  }
}

template <class T>
void SYCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  auto _a = d_a.template get_access<access::read>();
  auto _b = d_b.template get_access<access::read>();
  auto _c = d_c.template get_access<access::read>();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = _a[i];
    b[i] = _b[i];
    c[i] = _c[i];
  }
}

void listDevices(void)
{
  // TODO: Get actual list of devices
  std::cout << std::endl;
  std::cout << "Devices:" << std::endl;
  std::cout << "0: " << "triSYCL" << std::endl;
  std::cout << std::endl;
}

std::string getDeviceName(const int device)
{
  // TODO: Implement properly
  return "triSYCL";
}

std::string getDeviceDriver(const int device)
{
  // TODO: Implement properly
  return "triSCYL";
}


template class SYCLStream<float>;
template class SYCLStream<double>;
