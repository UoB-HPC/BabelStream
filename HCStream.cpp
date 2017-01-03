// Copyright (c) 2015-16 Peter Steinbach, MPI CBG Scientific Computing Facility
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include <codecvt>
#include <vector>
#include <locale>


#include "HCStream.h"
//#include "hc.hpp"

#define TBSIZE 1024

std::string getDeviceName(const hc::accelerator& _acc)
{
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  std::string value = converter.to_bytes(_acc.get_description());
  return value;
}

void listDevices(void)
{
  // Get number of devices
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  
  // Print device names
  if (accs.empty())
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < accs.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(accs[i]) << std::endl;
    }
    std::cout << std::endl;
  }
}

// void check_error(void)
// {
//   hipError_t err = hipGetLastError();
//   if (err != hipSuccess)
//   {
//     std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
//     exit(err);
//   }
// }

template <class T>
HCStream<T>::HCStream(const unsigned int ARRAY_SIZE, const int device_index):
  array_size(ARRAY_SIZE),
  d_a(ARRAY_SIZE),
  d_b(ARRAY_SIZE),
  d_c(ARRAY_SIZE)
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // // Set device
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  auto current = accs[device_index];
  
  std::cout << "Using HC device " << getDeviceName(current) << std::endl;
  
  // // The array size must be divisible by TBSIZE for kernel launches
  // if (ARRAY_SIZE % TBSIZE != 0)
  // {
  //   std::stringstream ss;
  //   ss << "Array size must be a multiple of " << TBSIZE;
  //   throw std::runtime_error(ss.str());
  // }

  // // Set device
  // int count;
  // hipGetDeviceCount(&count);
  // check_error();
  // if (device_index >= count)
  //   throw std::runtime_error("Invalid device index");
  // hipSetDevice(device_index);
  // check_error();

  // // Print out device information
  // std::cout << "Using HIP device " << getDeviceName(device_index) << std::endl;
  // std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  // array_size = ARRAY_SIZE;

  // // Check buffers fit on the device
  // hipDeviceProp_t props;
  // hipGetDeviceProperties(&props, 0);
  // if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
  //   throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // // Create device buffers
  // hipMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  // check_error();
  // hipMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  // check_error();
  // hipMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  // check_error();

  
}


template <class T>
HCStream<T>::~HCStream()
{
}

template <class T>
void HCStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  hc::copy(a.cbegin(),a.cend(),d_a);
  hc::copy(b.cbegin(),b.cend(),d_b);
  hc::copy(c.cbegin(),c.cend(),d_c);
}

template <class T>
void HCStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
  hc::copy(d_a,a.begin());
  hc::copy(d_b,b.begin());
  hc::copy(d_c,c.begin());
}


template <class T>
void HCStream<T>::copy()
{
  try{
  // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> i) [[hc]] {
								  d_c[i] = d_a[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::mul()
{
  const T scalar = 0.3;
  try{
  // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> i) [[hc]] {
								  d_b[i] = scalar*d_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::add()
{
  try{
    // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> i) [[hc]] {
								  d_c[i] = d_a[i]+d_b[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::triad()
{
  const T scalar = 0.3;
  try{
    // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> i) [[hc]] {
								  d_a[i] = d_b[i] + scalar*d_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << e.what() << std::endl;
    throw;
  }
}

template class HCStream<float>;
template class HCStream<double>;
