
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
  : array_size(array_size)
{
  if (!cached)
    getDeviceList();

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");

  sycl::device dev = devices[device_index];
  hostA.resize(array_size);
  hostB.resize(array_size);
  hostC.resize(array_size);

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

  auto async_handler = [&](sycl::exception_list l)
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
  };
  queue = std::make_unique<sycl::queue>(dev, async_handler, sycl::property::queue::in_order());

  // Allocate memory
#if defined(PAGEFAULT)
  a = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  b = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  c = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  sum = (T*)aligned_alloc(ALIGNMENT, ALIGNMENT);

#else
  use_shared_alloc =
        queue->get_device().has(sycl::aspect::usm_shared_allocations);
  a = use_shared_alloc ? sycl::malloc_shared<T>(array_size, *queue) :
                         sycl::malloc_device<T>(array_size, *queue);
  b = use_shared_alloc ? sycl::malloc_shared<T>(array_size, *queue) :
                         sycl::malloc_device<T>(array_size, *queue);
  c = use_shared_alloc ? sycl::malloc_shared<T>(array_size, *queue) :
                         sycl::malloc_device<T>(array_size, *queue);
  sum = use_shared_alloc ? sycl::malloc_shared<T>(1, *queue) :
                           sycl::malloc_device<T>(1, *queue);
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
    cgh.parallel_for(sycl::range<1>{array_size}, [c=c,a=a](sycl::id<1> idx)
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
    cgh.parallel_for(sycl::range<1>{array_size}, [=,b=b,c=c](sycl::id<1> idx)
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
    cgh.parallel_for(sycl::range<1>{array_size}, [c=c,a=a,b=b](sycl::id<1> idx)
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
    cgh.parallel_for(sycl::range<1>{array_size}, [=,a=a,b=b,c=c](sycl::id<1> idx)
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
    cgh.parallel_for(sycl::range<1>{array_size}, [=,a=a,b=b,c=c](sycl::id<1> idx)
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
      // AdaptiveCpp doesn't sypport the initialize_to_identity property yet
#if defined(__HIPSYCL__) || defined(__OPENSYCL__) || defined(__ADAPTIVECPP__)
      sycl::reduction(sum, sycl::plus<T>()),
#else
      sycl::reduction(sum, sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
#endif
      [a=a,b=b](sycl::id<1> idx, auto& sum)
      {
        sum += a[idx] * b[idx];
      });
  });
  if (use_shared_alloc) {
    queue->wait();
    return *sum;
  }

  T hostSum;
  queue->copy(sum, &hostSum, 1);
  queue->wait();
  return hostSum;
}

template <class T>
void SYCLStream<T>::init_arrays(T initA, T initB, T initC)
{
#if defined(PAGEFAULT)
  for (int i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
#else
  queue->submit([&](sycl::handler &cgh)
  {
    cgh.parallel_for(sycl::range<1>{array_size}, [=,a=a,b=b,c=c](sycl::id<1> idx)
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
  if (use_shared_alloc) {
    h_a = &a[0];
    h_b = &b[0];
    h_c = &c[0];
  } else {
    queue->copy(a, hostA.data(), array_size);
    queue->copy(b, hostB.data(), array_size);
    queue->copy(c, hostC.data(), array_size);
    queue->wait();

    h_a = hostA.data();
    h_b = hostB.data();
    h_c = hostC.data();
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
