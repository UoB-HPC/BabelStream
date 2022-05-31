// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "HIPStream.h"
#include "hip/hip_runtime.h"

#define TBSIZE 1024


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
  : array_size{ARRAY_SIZE},
    block_count(array_size / (TBSIZE * elements_per_lane))
{

  std::cerr << "Elements per lane: " << elements_per_lane << std::endl;
  std::cerr << "Chunks per block: " << chunks_per_block << std::endl;
  // The array size must be divisible by total number of elements
  // moved per block for kernel launches
  if (ARRAY_SIZE % (TBSIZE * elements_per_lane) != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of elements operated on per block ("
       << TBSIZE * elements_per_lane
       << ").";
    throw std::runtime_error(ss.str());
  }
  std::cerr << "block count " << block_count << std::endl;

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
  hipHostMalloc(&sums, sizeof(T) * block_count, hipHostMallocNonCoherent);
  check_error();

  // Check buffers fit on the device
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < std::size_t{3}*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  hipMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
}


template <class T>
HIPStream<T>::~HIPStream()
{
  hipHostFree(sums);
  check_error();

  hipFree(d_a);
  check_error();
  hipFree(d_b);
  check_error();
  hipFree(d_c);
  check_error();
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
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

template <size_t elements_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__
void copy_kernel(const T * __restrict a, T * __restrict c)
{
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t j = 0; j < elements_per_lane; ++j)
    c[gidx + j] = a[gidx + j];
}

template <class T>
void HIPStream<T>::copy()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel<elements_per_lane, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__
void mul_kernel(T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t j = 0; j < elements_per_lane; ++j)
    b[gidx + j] = scalar * c[gidx + j];
}

template <class T>
void HIPStream<T>::mul()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<elements_per_lane, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__
void add_kernel(const T * __restrict a, const T * __restrict b, T * __restrict c)
{
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t j = 0; j < elements_per_lane; ++j)
    c[gidx + j] = a[gidx + j] + b[gidx + j];
}

template <class T>
void HIPStream<T>::add()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<elements_per_lane, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__
void triad_kernel(T * __restrict a, const T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t j = 0; j < elements_per_lane; ++j)
    a[gidx + j] = b[gidx + j] + scalar * c[gidx + j];
}

template <class T>
void HIPStream<T>::triad()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel<elements_per_lane, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__ void nstream_kernel(T * __restrict a, const T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t j = 0; j < elements_per_lane; ++j)
    a[gidx + j] += b[gidx + j] + scalar * c[gidx + j];
}

template <class T>
void HIPStream<T>::nstream()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(nstream_kernel<elements_per_lane, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, typename T>
__launch_bounds__(TBSIZE)
__global__ void dot_kernel(const T * __restrict a, const T * __restrict b, T * __restrict sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  const size_t local_i = threadIdx.x;
  size_t i = blockDim.x * blockIdx.x + local_i;

  tb_sum[local_i] = 0.0;
  for (size_t j = 0; j < elements_per_lane && i < array_size; ++j, i += blockDim.x*gridDim.x)
    tb_sum[local_i] += a[i] * b[i];

  for (size_t offset = blockDim.x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[blockIdx.x] = tb_sum[local_i];
}

template <class T>
T HIPStream<T>::dot()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(dot_kernel<elements_per_lane, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_b, sums, array_size);
  check_error();
  hipDeviceSynchronize();
  check_error();

  T sum = 0.0;
  for (int i = 0; i < block_count; i++)
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
