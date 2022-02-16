
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStreamUSM.h"

#include <iostream>

using namespace cl::sycl;

// Cache list of devices
bool cached = false;
std::vector<device> devices;
void getDeviceList(void);

template <class T>
SYCLStream<T>::SYCLStream(const int ARRAY_SIZE, const int device_index)
{
  if (!cached)
    getDeviceList();

  array_size = ARRAY_SIZE;

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device dev = devices[device_index];

  // Check device can support FP64 if needed
  if (sizeof(T) == sizeof(double))
  {
    if (dev.get_info<info::device::double_fp_config>().size() == 0) {
      throw std::runtime_error("Device does not support double precision, please use --float");
    }
  }

  // Determine sensible dot kernel NDRange configuration
  if (dev.is_cpu())
  {
    dot_num_groups = dev.get_info<info::device::max_compute_units>();
    dot_wgsize     = dev.get_info<info::device::native_vector_width_double>() * 2;
  }
  else
  {
    dot_num_groups = dev.get_info<info::device::max_compute_units>() * 4;
    dot_wgsize     = dev.get_info<info::device::max_work_group_size>();
  }

  // Print out device information
  std::cout << "Using SYCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;

  queue = new cl::sycl::queue(dev, cl::sycl::async_handler{[&](cl::sycl::exception_list l)
  {
    bool error = false;
    for(auto e: l)
    {
      try
      {
        std::rethrow_exception(e);
      }
      catch (cl::sycl::exception e)
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
  
  // Create device memory
  d_a = static_cast<T *>(malloc_device(array_size * sizeof(T), *queue));
  d_b = static_cast<T *>(malloc_device(array_size * sizeof(T), *queue));
  d_c = static_cast<T *>(malloc_device(array_size * sizeof(T), *queue));
  d_sum = static_cast<T *>(malloc_device(dot_num_groups * sizeof(T), *queue));

  // create host copy of partial sums
  h_sum = static_cast<T *>(malloc(dot_num_groups * sizeof(T)));
}

template <class T>
SYCLStream<T>::~SYCLStream()
{
  sycl::free(d_a, *queue);
  sycl::free(d_b, *queue);
  sycl::free(d_c, *queue);
  sycl::free(d_sum, *queue);
  free(h_sum);
  delete queue;
  devices.clear();
}

template <class T>
void SYCLStream<T>::copy()
{
  T *l_c = d_c;
  const T *l_a = d_a;
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<copy_kernel>(range<1>{array_size}, [=](id<1> idx)
    {
      l_c[idx] = l_a[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::mul()
{
  const T scalar = startScalar;
  T *l_b = d_b;
  const T *l_c = d_c;
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<mul_kernel>(range<1>{array_size}, [=](id<1> idx)
    {
      l_b[idx] = scalar * l_c[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::add()
{
  const T *l_a = d_a;
  const T *l_b = d_b;
  T *l_c = d_c;
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<add_kernel>(range<1>{array_size}, [=](id<1> idx)
    {
      l_c[idx] = l_a[idx] + l_b[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::triad()
{
  const T scalar = startScalar;
  T *l_a = d_a;
  const T *l_b = d_b;
  const T *l_c = d_c;
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<triad_kernel>(range<1>{array_size}, [=](id<1> idx)
    {
      l_a[idx] = l_b[idx] + scalar * l_c[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::nstream()
{
  const T scalar = startScalar;
  T *l_a = d_a;
  const T *l_b = d_b;
  const T *l_c = d_c;
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<nstream_kernel>(range<1>{array_size}, [=](id<1> idx)
    {
      l_a[idx] += l_b[idx] + scalar * l_c[idx];
    });
  });
  queue->wait();
}

template <class T>
T SYCLStream<T>::dot()
{
  const T *l_a = d_a;
  const T *l_b = d_b;
  T *l_sum = d_sum;
  queue->submit([&](handler &cgh)
  {
    auto wg_sum = accessor<T, 1, 
        access::mode::read_write,
        access::target::local>(range<1>(dot_wgsize), cgh);

    size_t N = array_size;
    cgh.parallel_for<dot_kernel>(nd_range<1>(
        dot_num_groups*dot_wgsize, dot_wgsize),
        [=](nd_item<1> item)
    {
      size_t i = item.get_global_id(0);
      size_t li = item.get_local_id(0);
      size_t global_size = item.get_global_range()[0];

      wg_sum[li] = 0.0;
      for (; i < N; i += global_size)
        wg_sum[li] += l_a[i] * l_b[i];

      size_t local_size = item.get_local_range()[0];
      for (int offset = local_size / 2; offset > 0; offset /= 2)
      {
        item.barrier(cl::sycl::access::fence_space::local_space);
        if (li < offset)
          wg_sum[li] += wg_sum[li + offset];
      }

      if (li == 0)
        l_sum[item.get_group(0)] = wg_sum[0];
    });
  });

  queue->memcpy(h_sum, l_sum, sizeof(T) * dot_num_groups);
  queue->wait();

  T sum = 0.0;
  for (int i = 0; i < dot_num_groups; i++)
  {
    sum += h_sum[i];
  }

  return sum;
}

template <class T>
void SYCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  T *l_a = d_a;
  T *l_b = d_b;
  T *l_c = d_c;
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<init_kernel>(range<1>{array_size}, [=](item<1> item)
    {
      auto id = item.get_id(0);
      l_a[id] = initA;
      l_b[id] = initB;
      l_c[id] = initC;
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  const T *l_a = d_a;
  const T *l_b = d_b;
  const T *l_c = d_c;
  queue->memcpy(a.data(), l_a, sizeof(T) * array_size);
  queue->memcpy(b.data(), l_b, sizeof(T) * array_size);
  queue->memcpy(c.data(), l_c, sizeof(T) * array_size);
  queue->wait();
}

void getDeviceList(void)
{
  // Ask SYCL runtime for all devices in system
  devices = cl::sycl::device::get_devices();
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
template class SYCLStream<float>;
template class SYCLStream<double>;
