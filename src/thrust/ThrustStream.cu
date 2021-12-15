// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "ThrustStream.h"
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

static inline void synchronise()
{
// rocThrust doesn't synchronise between thrust calls
#if defined(THRUST_DEVICE_SYSTEM_HIP) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
  hipDeviceSynchronize();
#endif
}

template <class T>
ThrustStream<T>::ThrustStream(const int ARRAY_SIZE, int device)
    : array_size{ARRAY_SIZE}, a(array_size), b(array_size), c(array_size) {
  std::cout << "Using CUDA device: " << getDeviceName(device) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device) << std::endl;
  std::cout << "Thrust version: " << THRUST_VERSION << std::endl;

#if THRUST_DEVICE_SYSTEM == 0
  // as per Thrust docs, 0 is reserved for undefined backend
  std::cout << "Thrust backend: undefined" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  std::cout << "Thrust backend: CUDA" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
  std::cout << "Thrust backend: OMP" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
  std::cout << "Thrust backend: TBB" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
  std::cout << "Thrust backend: CPP" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
  std::cout << "Thrust backend: TBB" << std::endl;
#else

#if defined(THRUST_DEVICE_SYSTEM_HIP) && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_HIP
  std::cout << "Thrust backend: HIP" << std::endl;
#else
  std::cout << "Thrust backend: " << THRUST_DEVICE_SYSTEM << "(unknown)" << std::endl;
#endif

#endif

}

template <class T>
void ThrustStream<T>::init_arrays(T initA, T initB, T initC)
{
  thrust::fill(a.begin(), a.end(), initA);
  thrust::fill(b.begin(), b.end(), initB);
  thrust::fill(c.begin(), c.end(), initC);
  synchronise();
}

template <class T>
void ThrustStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  thrust::copy(a.begin(), a.end(), h_a.begin());
  thrust::copy(b.begin(), b.end(), h_b.begin());
  thrust::copy(c.begin(), c.end(), h_c.begin());
}

template <class T>
void ThrustStream<T>::copy()
{
  thrust::copy(a.begin(), a.end(),c.begin());
  synchronise();
}

template <class T>
void ThrustStream<T>::mul()
{
  const T scalar = startScalar;
  thrust::transform(
      c.begin(),
      c.end(),
      b.begin(),
      [=] __device__ __host__ (const T &ci){
        return ci * scalar;
      }
  );
  synchronise();
}

template <class T>
void ThrustStream<T>::add()
{
  thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end())),
      c.begin(),
      thrust::make_zip_function(
          [] __device__ __host__ (const T& ai, const T& bi){
            return ai + bi;
          })
  );
  synchronise();
}

template <class T>
void ThrustStream<T>::triad()
{
  const T scalar = startScalar;
  thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(b.begin(), c.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(b.end(), c.end())),
      a.begin(),
      thrust::make_zip_function(
          [=] __device__ __host__ (const T& bi, const T& ci){
            return bi + scalar * ci;
          })
  );
  synchronise();
}

template <class T>
void ThrustStream<T>::nstream()
{
  const T scalar = startScalar;
  thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end())),
      a.begin(),
      thrust::make_zip_function(
          [=] __device__ __host__ (const T& ai, const T& bi, const T& ci){
            return ai + bi + scalar * ci;
          })
  );
  synchronise();
}

template <class T>
T ThrustStream<T>::dot()
{
  return thrust::inner_product(a.begin(), a.end(), b.begin(), T{});
}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA || \
    (defined(THRUST_DEVICE_SYSTEM_HIP) && THRUST_DEVICE_SYSTEM_HIP == THRUST_DEVICE_SYSTEM)

#ifdef __NVCC__
#define IMPL_FN__(fn) cuda ## fn
#define IMPL_TYPE__(tpe) cuda ## tpe
#elif defined(__HIP_PLATFORM_HCC__)
#define IMPL_FN__(fn) hip ## fn
#define IMPL_TYPE__(tpe) hip ## tpe ## _t
#else
# error Unsupported compiler for Thrust
#endif

void check_error(void)
{
  IMPL_FN__(Error_t) err =  IMPL_FN__(GetLastError());
  if (err !=  IMPL_FN__(Success))
  {
    std::cerr << "Error: " <<  IMPL_FN__(GetErrorString(err)) << std::endl;
    exit(err);
  }
}

void listDevices(void)
{
  // Get number of devices
  int count;
  IMPL_FN__(GetDeviceCount(&count));
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}

std::string getDeviceName(const int device)
{
  IMPL_TYPE__(DeviceProp) props = {};
  IMPL_FN__(GetDeviceProperties(&props, device));
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  IMPL_FN__(SetDevice(device));
  check_error();
  int driver;
  IMPL_FN__(DriverGetVersion(&driver));
  check_error();
  return std::to_string(driver);
}

#undef IMPL_FN__
#undef IMPL_TPE__

#else

void listDevices(void)
{
  std::cout << "0: CPU" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("(device name unavailable)");
}

std::string getDeviceDriver(const int)
{
  return std::string("(device driver unavailable)");
}

#endif

template class ThrustStream<float>;
template class ThrustStream<double>;

