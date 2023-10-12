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
#if defined(MANAGED)
    std::cout << "Memory: MANAGED" << std::endl;
#elif defined(PAGEFAULT)
    std::cout << "Memory: PAGEFAULT" << std::endl;
#else
    std::cout << "Memory: DEFAULT" << std::endl;
#endif

  array_size = ARRAY_SIZE;
  // Round dot_num_blocks up to next multiple of (TBSIZE * dot_elements_per_lane)
  dot_num_blocks = (array_size + (TBSIZE * dot_elements_per_lane - 1)) / (TBSIZE * dot_elements_per_lane);

  size_t array_bytes = sizeof(T);
  array_bytes *= ARRAY_SIZE;
  size_t total_bytes = array_bytes * 3;

  // Allocate the host array for partial sums for dot kernels using hipHostMalloc.
  // This creates an array on the host which is visible to the device. However, it requires
  // synchronization (e.g. hipDeviceSynchronize) for the result to be available on the host
  // after it has been passed through to a kernel.
  hipHostMalloc(&sums, sizeof(T) * dot_num_blocks, hipHostMallocNonCoherent);
  check_error();

  // Check buffers fit on the device
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < std::size_t{3}*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

 // Create device buffers
#if defined(MANAGED)
  hipMallocManaged(&d_a, array_bytes);
  check_error();
  hipMallocManaged(&d_b, array_bytes);
  check_error();
  hipMallocManaged(&d_c, array_bytes);
  check_error();
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(array_bytes);
  d_b = (T*)malloc(array_bytes);
  d_c = (T*)malloc(array_bytes);
#else
  hipMalloc(&d_a, array_bytes);
  check_error();
  hipMalloc(&d_b, array_bytes);
  check_error();
  hipMalloc(&d_c, array_bytes);
  check_error();
#endif
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
  init_kernel<T><<<dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0>>>(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <class T>
void HIPStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{

  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
    hipDeviceSynchronize();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
#endif
}

template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  c[i] = a[i];
}

template <class T>
void HIPStream<T>::copy()
{
  copy_kernel<T><<<dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0>>>(d_a, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  b[i] = scalar * c[i];
}

template <class T>
void HIPStream<T>::mul()
{
  mul_kernel<T><<<dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0>>>(d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  c[i] = a[i] + b[i];
}

template <class T>
void HIPStream<T>::add()
{
  add_kernel<T><<<dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0>>>(d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  a[i] = b[i] + scalar * c[i];
}

template <class T>
void HIPStream<T>::triad()
{
  triad_kernel<T><<<dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0>>>(d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  a[i] += b[i] + scalar * c[i];
}

template <class T>
void HIPStream<T>::nstream()
{
  nstream_kernel<T><<<dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0>>>(d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  const size_t local_i = threadIdx.x;
  size_t i = blockDim.x * blockIdx.x + local_i;

  tb_sum[local_i] = {};
  for (; i < array_size; i += blockDim.x*gridDim.x)
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
  dot_kernel<T><<<dim3(dot_num_blocks), dim3(TBSIZE), 0, 0>>>(d_a, d_b, sums, array_size);
  check_error();
  hipDeviceSynchronize();
  check_error();

  T sum{};
  for (int i = 0; i < dot_num_blocks; i++)
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
