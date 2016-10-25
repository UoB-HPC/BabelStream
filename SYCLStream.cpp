
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream.h"

#include <iostream>

using namespace cl::sycl;

#define WGSIZE 256

#define DOT_WGSIZE 256
#define DOT_NUM_GROUPS 256

// Cache list of devices
bool cached = false;
std::vector<device> devices;
void getDeviceList(void);

template <class T>
SYCLStream<T>::SYCLStream(const unsigned int ARRAY_SIZE, const int device_index)
{
  if (!cached)
    getDeviceList();

  // The array size must be divisible by WGSIZE
  if (ARRAY_SIZE % WGSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << WGSIZE;
    throw std::runtime_error(ss.str());
  }

  array_size = ARRAY_SIZE;

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device dev = devices[device_index];

  // Print out device information
  std::cout << "Using SYCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  queue = new cl::sycl::queue(dev);

  // Create buffers
  d_a = new buffer<T>(array_size);
  d_b = new buffer<T>(array_size);
  d_c = new buffer<T>(array_size);
  d_sum = new buffer<T>(DOT_NUM_GROUPS);
}

template <class T>
SYCLStream<T>::~SYCLStream()
{
  delete d_a;
  delete d_b;
  delete d_c;
  delete d_sum;

  delete queue;
}

template <class T>
void SYCLStream<T>::copy()
{
  queue->submit([&](handler &cgh)
  {
    auto ka = d_a->template get_access<access::mode::read>(cgh);
    auto kc = d_c->template get_access<access::mode::write>(cgh);
    cgh.parallel_for<class copy>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      kc[item.get_global()] = ka[item.get_global()];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::mul()
{
  const T scalar = startScalar;
  queue->submit([&](handler &cgh)
  {
    auto kb = d_b->template get_access<access::mode::write>(cgh);
    auto kc = d_c->template get_access<access::mode::read>(cgh);
    cgh.parallel_for<class mul>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      kb[item.get_global()] = scalar * kc[item.get_global()];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::add()
{
  queue->submit([&](handler &cgh)
  {
    auto ka = d_a->template get_access<access::mode::read>(cgh);
    auto kb = d_b->template get_access<access::mode::read>(cgh);
    auto kc = d_c->template get_access<access::mode::write>(cgh);
    cgh.parallel_for<class add>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      kc[item.get_global()] = ka[item.get_global()] + kb[item.get_global()];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::triad()
{
  const T scalar = startScalar;
  queue->submit([&](handler &cgh)
  {
    auto ka = d_a->template get_access<access::mode::write>(cgh);
    auto kb = d_b->template get_access<access::mode::read>(cgh);
    auto kc = d_c->template get_access<access::mode::read>(cgh);
    cgh.parallel_for<class triad>(nd_range<1>{array_size, WGSIZE}, [=](nd_item<1> item)
    {
      ka[item.get_global()] = kb[item.get_global()] + scalar * kc[item.get_global()];
    });
  });
  queue->wait();
}

template <class T>
T SYCLStream<T>::dot()
{
  queue->submit([&](handler &cgh)
  {
    auto ka   = d_a->template get_access<access::mode::read>(cgh);
    auto kb   = d_b->template get_access<access::mode::read>(cgh);
    auto ksum = d_sum->template get_access<access::mode::write>(cgh);

    auto wg_sum = accessor<T, 1, access::mode::read_write, access::target::local>(range<1>(DOT_WGSIZE), cgh);

    size_t N = array_size;

    cgh.parallel_for<class dot>(nd_range<1>(DOT_NUM_GROUPS*DOT_WGSIZE, DOT_WGSIZE), [=](nd_item<1> item)
    {
      size_t i = item.get_global(0);
      size_t li = item.get_local(0);
      wg_sum[li] = 0.0;
      for (; i < N; i += item.get_global_range()[0])
        wg_sum[li] += ka[i] * kb[i];

      for (int offset = item.get_local_range()[0]; offset > 0; offset /= 2)
      {
        item.barrier(cl::sycl::access::fence_space::local_space);
        if (li < offset)
          wg_sum[li] += wg_sum[li + offset];
      }

      if (li == 0)
        ksum[item.get_group(0)] = wg_sum[0];
    });
  });

  T sum = 0.0;
  auto h_sum = d_sum->template get_access<access::mode::read, access::target::host_buffer>();
  for (int i = 0; i < DOT_NUM_GROUPS; i++)
  {
    sum += h_sum[i];
  }

  return sum;
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

void getDeviceList(void)
{
  // Get list of platforms
  std::vector<platform> platforms = platform::get_platforms();

  // Enumerate devices
  for (unsigned i = 0; i < platforms.size(); i++)
  {
    std::vector<device> plat_devices = platforms[i].get_devices();
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;

  if (device < devices.size())
  {
    name = devices[device].get_info<info::device::name>();
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;
}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    driver = devices[device].get_info<info::device::driver_version>();
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}


// TODO: Fix kernel names to allow multiple template specializations
//template class SYCLStream<float>;
template class SYCLStream<double>;
