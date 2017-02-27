// Copyright (c) 2015-16 Peter Steinbach, MPI CBG Scientific Computing Facility
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include <codecvt>
#include <vector>
#include <locale>
#include <numeric>

#include "HCStream.h"

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

  hc::accelerator::set_default(current.get_device_path());

  std::cout << "Using HC device " << getDeviceName(current) << std::endl;

}


template <class T>
HCStream<T>::~HCStream()
{
}

template <class T>
void HCStream<T>::init_arrays(T _a, T _b, T _c)
{
  std::vector<T> temp(array_size,_a);
  hc::copy(temp.begin(), temp.end(),this->d_a);

  std::fill(temp.begin(), temp.end(),_b);
  hc::copy(temp.begin(), temp.end(),this->d_b);

  std::fill(temp.begin(), temp.end(),_c);
  hc::copy(temp.begin(), temp.end(),this->d_c);

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

  hc::array<T,1>& device_a = this->d_a;
  hc::array<T,1>& device_c = this->d_c;

  try{
  // launch a GPU kernel to compute the saxpy in parallel
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> index) [[hc]] {
                                  device_c[index] = device_a[index];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::mul()
{
  const T scalar = 0.3;
  hc::array<T,1>& device_b = this->d_b;
  hc::array<T,1>& device_c = this->d_c;

  try{
  // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> i) [[hc]] {
                                  device_b[i] = scalar*device_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::add()
{

  hc::array<T,1>& device_a = this->d_a;
  hc::array<T,1>& device_b = this->d_b;
  hc::array<T,1>& device_c = this->d_c;

  try{
    // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> i) [[hc]] {
                                  device_c[i] = device_a[i]+device_b[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::triad()
{
  const T scalar = 0.3;
  hc::array<T,1>& device_a = this->d_a;
  hc::array<T,1>& device_b = this->d_b;
  hc::array<T,1>& device_c = this->d_c;

  try{
    // launch a GPU kernel to compute the saxpy in parallel 
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
								, [&](hc::index<1> i) [[hc]] {
                                  device_a[i] = device_b[i] + scalar*device_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;
    throw;
  }
}

template <class T>
T HCStream<T>::dot()
{
  hc::array<T,1>& device_a = this->d_a;
  hc::array<T,1> product = this->d_b;

  T sum = static_cast<T>(0);

  try{
    // launch a GPU kernel to compute the saxpy in parallel
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
                                                                , [&](hc::index<1> i) [[hc]] {
                                                                  product[i] *= device_a[i];
                                                                });
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cout << __FILE__ << ":" << __LINE__ << "\t" << e.what() << std::endl;
    throw;
  }

  std::vector<T> h_product(array_size,sum);
  hc::copy(product,h_product.begin());

  sum = std::accumulate(h_product.begin(), h_product.end(),sum);

  return sum;
}


template class HCStream<float>;
template class HCStream<double>;
