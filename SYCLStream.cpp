
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream.h"

#include <iostream>

using namespace cl::sycl;

#define WGSIZE 64

template <class T>
SYCLStream<T>::SYCLStream(const unsigned int ARRAY_SIZE, const int device_index)
{
  array_size = ARRAY_SIZE;

  // Create buffers
  d_a = new buffer<T>(array_size);
  d_b = new buffer<T>(array_size);
  d_c = new buffer<T>(array_size);
}

template <class T>
SYCLStream<T>::~SYCLStream()
{
  delete d_a;
  delete d_b;
  delete d_c;
}

template <class T>
void SYCLStream<T>::copy()
{
  queue.submit([&](handler &cgh)
  {
    auto ka = d_a->template get_access<access::mode::read>(cgh);
    auto kc = d_c->template get_access<access::mode::write>(cgh);
    cgh.parallel_for<class copy>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      kc[item.get_global()] = ka[item.get_global()];
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
    auto kb = d_b->template get_access<access::mode::write>(cgh);
    auto kc = d_c->template get_access<access::mode::read>(cgh);
    cgh.parallel_for<class mul>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      kb[item.get_global()] = scalar * kc[item.get_global()];
    });
  });
  queue.wait();
}

template <class T>
void SYCLStream<T>::add()
{
  queue.submit([&](handler &cgh)
  {
    auto ka = d_a->template get_access<access::mode::read>(cgh);
    auto kb = d_b->template get_access<access::mode::read>(cgh);
    auto kc = d_c->template get_access<access::mode::write>(cgh);
    cgh.parallel_for<class add>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      kc[item.get_global()] = ka[item.get_global()] + kb[item.get_global()];
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
    auto ka = d_a->template get_access<access::mode::write>(cgh);
    auto kb = d_b->template get_access<access::mode::read>(cgh);
    auto kc = d_c->template get_access<access::mode::read>(cgh);
    cgh.parallel_for<class triad>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      ka[item.get_global()] = kb[item.get_global()] + scalar * kc[item.get_global()];
    });
  });
  queue.wait();
}

template <class T>
void SYCLStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  auto _a = d_a->template get_access<access::mode::write, access::target::host_buffer>();
  auto _b = d_b->template get_access<access::mode::write, access::target::host_buffer>();
  auto _c = d_c->template get_access<access::mode::write, access::target::host_buffer>();
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
  auto _a = d_a->template get_access<access::mode::read, access::target::host_buffer>();
  auto _b = d_b->template get_access<access::mode::read, access::target::host_buffer>();
  auto _c = d_c->template get_access<access::mode::read, access::target::host_buffer>();
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


// TODO: Fix kernel names to allow multiple template specializations
//template class SYCLStream<float>;
template class SYCLStream<double>;
