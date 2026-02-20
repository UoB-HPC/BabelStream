
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

  // Allocate memory
#if defined(PAGEFAULT)
  a = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  b = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  c = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  sum = (T*)aligned_alloc(ALIGNMENT, ALIGNMENT);

#elseif defined(SYCL2020ACC)
  d_a = sycl::buffer<T>{array_size};
  d_b = sycl::buffer<T>{array_size};
  d_c = sycl::buffer<T>{array_size};
  d_sum = sycl::buffer<T>{1};

#elif SYCL2020USM
  a = sycl::malloc_shared<T>(array_size, *queue);
  b = sycl::malloc_shared<T>(array_size, *queue);
  c = sycl::malloc_shared<T>(array_size, *queue);
  sum = sycl::malloc_shared<T>(1, *queue);

#else
  #error unimplemented
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
#ifdef SYCL2020USM
  sycl::free(a, *queue);
  sycl::free(b, *queue);
  sycl::free(c, *queue);
  sycl::free(sum, *queue);
#else
  #error unimplemented
#endif
}

template <class T>
void SYCLStream<T>::copy()
{
  queue->submit([&](sycl::handler &cgh)
  {
#ifdef SYCL2020ACC
    sycl::accessor a {d_a, cgh, sycl::read_only};
    sycl::accessor c {d_c, cgh, sycl::write_only};
#endif    
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
#ifdef SYCL2020ACC
    sycl::accessor b {d_b, cgh, sycl::write_only};
    sycl::accessor c {d_c, cgh, sycl::read_only};
#endif    
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
#ifdef SYCL2020ACC
    sycl::accessor a {d_a, cgh, sycl::read_only};
    sycl::accessor b {d_b, cgh, sycl::read_only};
    sycl::accessor c {d_c, cgh, sycl::write_only};
#endif    
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
#ifdef SYCL2020ACC    
    sycl::accessor a {d_a, cgh, sycl::write_only};
    sycl::accessor b {d_b, cgh, sycl::read_only};
    sycl::accessor c {d_c, cgh, sycl::read_only};
#endif    
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
#if SYCL2020ACC
    sycl::accessor a {d_a, cgh};
    sycl::accessor b {d_b, cgh, sycl::read_only};
    sycl::accessor c {d_c, cgh, sycl::read_only};
#endif    
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
#if SYCL2020ACC    
    sycl::accessor a {d_a, cgh, sycl::read_only};
    sycl::accessor b {d_b, cgh, sycl::read_only};
#endif
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
  queue->wait();
  return *sum;
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
#if SYCL2020ACC    
    sycl::accessor a {d_a, cgh, sycl::write_only, sycl::no_init};
    sycl::accessor b {d_b, cgh, sycl::write_only, sycl::no_init};
    sycl::accessor c {d_c, cgh, sycl::write_only, sycl::no_init};
#endif
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
#if SYCL2020ACC
  sycl::host_accessor a {d_a, sycl::read_only};
  sycl::host_accessor b {d_b, sycl::read_only};
  sycl::host_accessor c {d_c, sycl::read_only};
#endif  
  h_a = &a[0];
  h_b = &b[0];
  h_c = &c[0];
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
