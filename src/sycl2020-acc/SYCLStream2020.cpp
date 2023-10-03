
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
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
: array_size {ARRAY_SIZE},
  d_a {ARRAY_SIZE},
  d_b {ARRAY_SIZE},
  d_c {ARRAY_SIZE},
  d_sum {1}
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

  // No longer need list of devices
  devices.clear();
  cached = true;


}


template <class T>
void SYCLStream<T>::copy()
{
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {d_a, cgh, sycl::read_only};
    sycl::accessor kc {d_c, cgh, sycl::write_only};
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
    sycl::accessor kb {d_b, cgh, sycl::write_only};
    sycl::accessor kc {d_c, cgh, sycl::read_only};
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
    sycl::accessor ka {d_a, cgh, sycl::read_only};
    sycl::accessor kb {d_b, cgh, sycl::read_only};
    sycl::accessor kc {d_c, cgh, sycl::write_only};
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
    sycl::accessor ka {d_a, cgh, sycl::write_only};
    sycl::accessor kb {d_b, cgh, sycl::read_only};
    sycl::accessor kc {d_c, cgh, sycl::read_only};
    cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx)
    {
      ka[idx] = kb[idx] + scalar * kc[idx];
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
    sycl::accessor ka {d_a, cgh};
    sycl::accessor kb {d_b, cgh, sycl::read_only};
    sycl::accessor kc {d_c, cgh, sycl::read_only};
    cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx)
    {
      ka[idx] += kb[idx] + scalar * kc[idx];
    });
  });
  queue->wait();
}

template <class T>
T SYCLStream<T>::dot()
{

  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {d_a, cgh, sycl::read_only};
    sycl::accessor kb {d_b, cgh, sycl::read_only};

    cgh.parallel_for(sycl::range<1>{array_size},
       // Reduction object, to perform summation - initialises the result to zero
       // hipSYCL doesn't sypport the initialize_to_identity property yet
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
      sycl::reduction(d_sum. template get_access<sycl::access_mode::read_write>(cgh), sycl::plus<T>()),
#else
      sycl::reduction(d_sum, cgh, sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
#endif
      [=](sycl::id<1> idx, auto& sum)
      {
        sum += ka[idx] * kb[idx];
      });

  });

  // Get access on the host, and return a copy of the data (single number)
  // This will block until the result is available, so no need to wait on the queue.
  sycl::host_accessor result {d_sum, sycl::read_only};
  return result[0];

}

template <class T>
void SYCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {d_a, cgh, sycl::write_only, sycl::no_init};
    sycl::accessor kb {d_b, cgh, sycl::write_only, sycl::no_init};
    sycl::accessor kc {d_c, cgh, sycl::write_only, sycl::no_init};

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
  sycl::host_accessor _a {d_a, sycl::read_only};
  sycl::host_accessor _b {d_b, sycl::read_only};
  sycl::host_accessor _c {d_c, sycl::read_only};
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
