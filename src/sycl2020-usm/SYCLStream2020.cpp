
// Copyright (c) 2015-23 Tom Deakin, Simon McIntosh-Smith, and Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream2020.h"

#include <iostream>

#define ALIGNMENT (1024 * 1024 * 2)

// Cache list of devices
bool cached = false;
std::vector<sycl::device> devices;
void getDeviceList(void);

template <class T>
SYCLStream<T>::SYCLStream(BenchId bs, const intptr_t array_size, const int device_index,
                          T initA, T initB, T initC)
: array_size {size_t(array_size)}
{
  if (!cached)
    getDeviceList();

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");

  sycl::device dev = devices[device_index];

  // Print out device information
  std::cout << "Using SYCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  // Check device can support FP64 if needed
  if (sizeof(T) == sizeof(double))
  {
    if (!dev.has(sycl::aspect::fp64))
    {
      throw std::runtime_error("Device does not support double precision, please use --float");
    }
  }

  queue = std::make_unique<sycl::queue>(dev, sycl::async_handler{[&](sycl::exception_list l)
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

#if defined(PAGEFAULT)
  a = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  b = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  c = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  sum = (T*)aligned_alloc(ALIGNMENT, ALIGNMENT);
#else
  a = sycl::malloc_shared<T>(array_size, *queue);
  b = sycl::malloc_shared<T>(array_size, *queue);
  c = sycl::malloc_shared<T>(array_size, *queue);
  sum = sycl::malloc_shared<T>(1, *queue);
#endif

  // No longer need list of devices
  devices.clear();
  cached = true;

  init_arrays(initA, initB, initC);
}

template<class T>
SYCLStream<T>::~SYCLStream() {
#if defined(PAGEFAULT)
 free(a);
 free(b);
 free(c);
 free(sum);
#else
 sycl::free(a, *queue);
 sycl::free(b, *queue);
 sycl::free(c, *queue);
 sycl::free(sum, *queue);
#endif
}

template <class T>
void SYCLStream<T>::copy()
{
  queue->submit([&](sycl::handler &cgh)
  {
    cgh.parallel_for(sycl::range<1>{array_size}, [=, c = this->c, a = this->a](sycl::id<1> idx)
    {
      c[idx] = a[idx];
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
    cgh.parallel_for(sycl::range<1>{array_size}, [=, b = this->b, c = this->c](sycl::id<1> idx)
    {
      b[idx] = scalar * c[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::add()
{
  queue->submit([&](sycl::handler &cgh)
  {
    cgh.parallel_for(sycl::range<1>{array_size}, [=, c = this->c, a = this->a, b = this->b](sycl::id<1> idx)
    {
      c[idx] = a[idx] + b[idx];
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
    cgh.parallel_for(sycl::range<1>{array_size}, [=, a = this->a, b = this->b, c = this->c](sycl::id<1> idx)
    {
      a[idx] = b[idx] + scalar * c[idx];
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::nstream()
{
  const T scalar = startScalar;

  queue->submit([&](sycl::handler &cgh)
  {
    cgh.parallel_for(sycl::range<1>{array_size}, [=, a = this->a, b = this->b, c = this->c](sycl::id<1> idx)
    {
      a[idx] += b[idx] + scalar * c[idx];
    });
  });
  queue->wait();
}

template <class T>
T SYCLStream<T>::dot()
{
  // Zero the result explicitly as initialize_to_identity is not supported everywhere
  queue->fill(sum, T{}, 1).wait();
  queue->submit([&](sycl::handler &cgh)
  {
    cgh.parallel_for(sycl::range<1>{array_size},
      sycl::reduction(sum, sycl::plus<T>()),
      [a = this->a, b = this->b](sycl::id<1> idx, auto& sum)
      {
        sum += a[idx] * b[idx];
      });

  });
  queue->wait();
  return *sum;
}

template <class T>
void SYCLStream<T>::init_arrays(T initA, T initB, T initC)
{
#if defined(PAGEFAULT)
  for (size_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
#else
  queue->submit([&](sycl::handler &cgh)
  {
    cgh.parallel_for(sycl::range<1>{array_size}, [=, a = this->a, b = this->b, c = this->c](sycl::id<1> idx)
    {
      a[idx] = initA;
      b[idx] = initB;
      c[idx] = initC;
    });
  });

  queue->wait();
#endif
}

template <class T>
void SYCLStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
  queue->wait();
  h_a = a;
  h_b = b;
  h_c = c;
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
