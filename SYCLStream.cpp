
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream.h"

#include <iostream>

// Cache list of devices
bool cached = false;
std::vector<sycl::device> devices;
void getDeviceList(void);

template <class T>
SYCLStream<T>::SYCLStream(const int ARRAY_SIZE, const int device_index)
{
  if (!cached)
    getDeviceList();

  array_size = ARRAY_SIZE;

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");

  sycl::device dev = devices[device_index];

  // Determine sensible dot kernel NDRange configuration
  if (dev.is_cpu())
  {
    dot_num_groups = dev.get_info<sycl::info::device::max_compute_units>();
    dot_wgsize     = dev.get_info<sycl::info::device::native_vector_width_double>() * 2;
  }
  else
  {
    dot_num_groups = dev.get_info<sycl::info::device::max_compute_units>() * 4;
    dot_wgsize     = dev.get_info<sycl::info::device::max_work_group_size>();
  }

  // Print out device information
  std::cout << "Using SYCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;

  queue = new sycl::queue(dev, sycl::async_handler{[&](sycl::exception_list l)
  {
    bool error = false;
    for(auto e: l)
    {
      try
      {
        std::rethrow_exception(e);
      }
      catch (sycl::exception e)
      {
        std::cout << e.what();
        error = true;
      }
    }
    if(error)
    {
      throw std::runtime_error("SYCL errors detected");
    }
  }});
  
  // Create buffers
  d_a = new sycl::buffer<T>(array_size);
  d_b = new sycl::buffer<T>(array_size);
  d_c = new sycl::buffer<T>(array_size);
  d_sum = new sycl::buffer<T>(dot_num_groups);
}

template <class T>
SYCLStream<T>::~SYCLStream()
{
  delete d_a;
  delete d_b;
  delete d_c;
  delete d_sum;
  delete queue;
  devices.clear();
}

template <class T>
void SYCLStream<T>::copy()
{
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {*d_a, cgh, sycl::read_only};
    sycl::accessor kc {*d_c, cgh, sycl::write_only};
    cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx)
    {
      kc[idx] = ka[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::mul()
{
  const T scalar = startScalar;
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor kb {*d_b, cgh, sycl::write_only};
    sycl::accessor kc {*d_c, cgh, sycl::read_only};
    cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx)
    {
      kb[idx] = scalar * kc[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::add()
{
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {*d_a, cgh, sycl::read_only};
    sycl::accessor kb {*d_b, cgh, sycl::read_only};
    sycl::accessor kc {*d_c, cgh, sycl::write_only};
    cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx)
    {
      kc[idx] = ka[idx] + kb[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::triad()
{
  const T scalar = startScalar;
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {*d_a, cgh, sycl::write_only};
    sycl::accessor kb {*d_b, cgh, sycl::read_only};
    sycl::accessor kc {*d_c, cgh, sycl::read_only};
    cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx)
    {
      ka[idx] = kb[idx] + scalar * kc[idx];
    });
  });
  queue->wait();
}

template <class T>
T SYCLStream<T>::dot()
{
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {*d_a, cgh, sycl::read_only};
    sycl::accessor kb {*d_b, cgh, sycl::read_only};
    sycl::accessor ksum {*d_sum, cgh, sycl::write_only};

    //sycl::local_accessor wg_sum {sycl::range<1>(dot_wgsize), cgh};
    auto wg_sum = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>(sycl::range<1>(dot_wgsize), cgh);

    size_t N = array_size;
    cgh.parallel_for(sycl::nd_range<1>(dot_num_groups*dot_wgsize, dot_wgsize), [=](sycl::nd_item<1> item)
    {
      size_t i = item.get_global_id(0);
      size_t li = item.get_local_id(0);
      size_t global_size = item.get_global_range()[0];

      wg_sum[li] = 0.0;
      for (; i < N; i += global_size)
        wg_sum[li] += ka[i] * kb[i];

      size_t local_size = item.get_local_range()[0];
      for (int offset = local_size / 2; offset > 0; offset /= 2)
      {
        item.barrier(sycl::access::fence_space::local_space);
        if (li < offset)
          wg_sum[li] += wg_sum[li + offset];
      }

      if (li == 0)
        ksum[item.get_group(0)] = wg_sum[0];
    });
  });

  T sum = 0.0;
  auto h_sum = d_sum->template get_access<sycl::access::mode::read>();
  for (int i = 0; i < dot_num_groups; i++)
  {
    sum += h_sum[i];
  }

  return sum;
}

template <class T>
void SYCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  queue->submit([&](sycl::handler &cgh)
  {
    // TODO: could add the sycl::no_init property
    sycl::accessor ka {*d_a, cgh, sycl::write_only};
    sycl::accessor kb {*d_b, cgh, sycl::write_only};
    sycl::accessor kc {*d_c, cgh, sycl::write_only};
    cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx)
    {
      ka[idx] = initA;
      kb[idx] = initB;
      kc[idx] = initC;
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  auto _a = d_a->template get_access<sycl::access::mode::read>();
  auto _b = d_b->template get_access<sycl::access::mode::read>();
  auto _c = d_c->template get_access<sycl::access::mode::read>();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = _a[i];
    b[i] = _b[i];
    c[i] = _c[i];
  }
}

void getDeviceList(void)
{
  // Ask SYCL runtime for all devices in system
  devices = sycl::device::get_devices();
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
    name = devices[device].get_info<sycl::info::device::name>();
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
    driver = devices[device].get_info<sycl::info::device::driver_version>();
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}

template class SYCLStream<float>;
template class SYCLStream<double>;
