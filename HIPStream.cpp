// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "HIPStream.h"
#include "hip/hip_runtime.h"

#define TBSIZE 1024
#define DOT_NUM_BLOCKS 256

void check_error(void)
{
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
  {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
HIPStream<T>::HIPStream(const int ARRAY_SIZE, const int device_index)
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // Set device
  int count;
  hipGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  hipSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using HIP device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  array_size = ARRAY_SIZE;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(sizeof(T) * DOT_NUM_BLOCKS);

  // Check buffers fit on the device
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  hipMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();
}


template <class T>
HIPStream<T>::~HIPStream()
{
  free(sums);

  hipFree(d_a);
  check_error();
  hipFree(d_b);
  check_error();
  hipFree(d_c);
  check_error();
  hipFree(d_sum);
  check_error();
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void HIPStream<T>::init_arrays(T initA, T initB, T initC)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(init_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_b, d_c, initA, initB, initC);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <class T>
void HIPStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
  hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  c[i] = a[i];
}

template <class T>
void HIPStream<T>::copy()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  b[i] = scalar * c[i];
}

template <class T>
void HIPStream<T>::mul()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  c[i] = a[i] + b[i];
}

template <class T>
void HIPStream<T>::add()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = b[i] + scalar * c[i];
}

template <class T>
void HIPStream<T>::triad()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] += b[i] + scalar * c[i];
}

template <class T>
void HIPStream<T>::nstream()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(nstream_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  const size_t local_i = hipThreadIdx_x;

  tb_sum[local_i] = 0.0;
  for (; i < array_size; i += hipBlockDim_x*hipGridDim_x)
    tb_sum[local_i] += a[i] * b[i];

  for (int offset = hipBlockDim_x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[hipBlockIdx_x] = tb_sum[local_i];
}

template <class T>
T HIPStream<T>::dot()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(dot_kernel<T>), dim3(DOT_NUM_BLOCKS), dim3(TBSIZE), 0, 0, d_a, d_b, d_sum, array_size);
  check_error();

  hipMemcpy(sums, d_sum, DOT_NUM_BLOCKS*sizeof(T), hipMemcpyDeviceToHost);
  check_error();

  T sum = 0.0;
  for (int i = 0; i < DOT_NUM_BLOCKS; i++)
    sum += sums[i];

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  hipGetDeviceCount(&count);
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
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  hipSetDevice(device);
  check_error();
  int driver;
  hipDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class HIPStream<float>;
template class HIPStream<double>;
