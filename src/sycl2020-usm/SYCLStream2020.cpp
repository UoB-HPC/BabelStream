
// Copyright (c) 2015-23 Tom Deakin, Simon McIntosh-Smith, and Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream2020.h"

#include <iostream>

// Cache list of devices
bool cached = false;
std::vector<sycl::device> devices;
void getDeviceList(void);

template <class T>
SYCLStream<T>::SYCLStream(const size_t ARRAY_SIZE, const int device_index)
: array_size {ARRAY_SIZE}
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

  a = sycl::malloc_shared<T>(array_size, *queue);
  b = sycl::malloc_shared<T>(array_size, *queue);
  c = sycl::malloc_shared<T>(array_size, *queue);
  sum = sycl::malloc_shared<T>(1, *queue);

  // No longer need list of devices
  devices.clear();
  cached = true;


}

template<class T>
SYCLStream<T>::~SYCLStream() {
 sycl::free(a, *queue);
 sycl::free(b, *queue);
 sycl::free(c, *queue);
 sycl::free(sum, *queue);
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
  queue->submit([&](sycl::handler &cgh)
  {
    cgh.parallel_for(sycl::range<1>{array_size},
      // Reduction object, to perform summation - initialises the result to zero
      // hipSYCL doesn't sypport the initialize_to_identity property yet
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
      sycl::reduction(sum, sycl::plus<T>()),
#else
      sycl::reduction(sum, sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
#endif
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
}

template <class T>
void SYCLStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  for (int i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
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
