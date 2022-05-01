// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "HIPStream.h"
#include "hip/hip_runtime.h"

#define TBSIZE 1024

#ifdef NONTEMPORAL
template<typename T>
__device__ __forceinline__ T load(const T& ref)
{
  return __builtin_nontemporal_load(&ref);
}

template<typename T>
__device__ __forceinline__ void store(const T& value, T& ref)
{
  __builtin_nontemporal_store(value, &ref);
}
#else
template<typename T>
__device__ __forceinline__ T load(const T& ref)
{
  return ref;
}

template<typename T>
__device__ __forceinline__ void store(const T& value, T& ref)
{
  ref = value;
}
#endif

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
    block_count(array_size / (TBSIZE * elements_per_lane * chunks_per_block))
{

  std::cerr << "Elements per lane: " << elements_per_lane << std::endl;
  std::cerr << "Chunks per block: " << chunks_per_block << std::endl;
  // The array size must be divisible by total number of elements
  // moved per block for kernel launches
  if (ARRAY_SIZE % (TBSIZE * elements_per_lane * chunks_per_block) != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of elements operated on per block ("
       << TBSIZE * elements_per_lane * chunks_per_block
       << ").";
    throw std::runtime_error(ss.str());
  }
  std::cerr << "block count " << block_count << std::endl;

#ifdef NONTEMPORAL
  std::cerr << "Using non-temporal memory operations." << std::endl;
#endif

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
  sums = (T*)malloc(block_count*sizeof(T));

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
  hipMalloc(&d_sum, block_count*sizeof(T));
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

template <size_t elements_per_lane, size_t chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void copy_kernel(const T * __restrict a, T * __restrict c)
{
  const size_t dx = (blockDim.x * gridDim.x) * elements_per_lane;
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t i = 0; i != chunks_per_block; ++i)
  {
    for (size_t j = 0; j != elements_per_lane; ++j)
    {
      store(load(a[gidx + i * dx + j]), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
void HIPStream<T>::copy()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel<elements_per_lane, chunks_per_block, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, size_t chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void mul_kernel(T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const size_t dx = (blockDim.x * gridDim.x) * elements_per_lane;
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t i = 0; i != chunks_per_block; ++i)
  {
    for (size_t j = 0; j != elements_per_lane; ++j)
    {
      store(scalar * load(c[gidx + i * dx + j]), b[gidx + i * dx + j]);
    }
  }
}

template <class T>
void HIPStream<T>::mul()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<elements_per_lane, chunks_per_block, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, size_t chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void add_kernel(const T * __restrict a, const T * __restrict b, T * __restrict c)
{
  const size_t dx = (blockDim.x * gridDim.x) * elements_per_lane;
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t i = 0; i != chunks_per_block; ++i)
  {
    for (size_t j = 0; j != elements_per_lane; ++j)
    {
      store(load(a[gidx + i * dx + j]) + load(b[gidx + i * dx + j]), c[gidx + i * dx + j]);
    }
  }
}

template <class T>
void HIPStream<T>::add()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<elements_per_lane, chunks_per_block, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <size_t elements_per_lane, size_t chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
void triad_kernel(T * __restrict a, const T * __restrict b, const T * __restrict c)
{
  const T scalar = startScalar;
  const size_t dx = (blockDim.x * gridDim.x) * elements_per_lane;
  const size_t gidx = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_lane;
  for (size_t i = 0; i != chunks_per_block; ++i)
  {
    for (size_t j = 0; j != elements_per_lane; ++j)
    {
      store(load(b[gidx + i * dx + j]) + scalar * load(c[gidx + i * dx + j]), a[gidx + i * dx + j]);
    }
  }
}

template <class T>
void HIPStream<T>::triad()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel<elements_per_lane, chunks_per_block, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_b, d_c);
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

template<unsigned int n = TBSIZE>
struct Reducer
{
  template<typename I>
  __device__
  static
  void reduce(I it) noexcept
  {
    if (n == 1) return;

#if defined(__HIP_PLATFORM_NVCC__)
    constexpr unsigned int warpSize = 32;
#endif
    constexpr bool is_same_warp{n <= warpSize * 2};
    if (static_cast<int>(threadIdx.x) < n/2)
    {
      it[threadIdx.x] += it[threadIdx.x + n/2];
    }
    is_same_warp ? __threadfence_block() : __syncthreads();
    Reducer<n/2>::reduce(it);
  }
};

template<>
struct Reducer<1u> {
  template<typename I>
  __device__
  static
  void reduce(I) noexcept
  {}
};

template <size_t elements_per_lane, size_t chunks_per_block, typename T>
__launch_bounds__(TBSIZE)
__global__
__global__ void dot_kernel(const T * __restrict a, const T * __restrict b, T * __restrict sum)
{
  __shared__ T tb_sum[TBSIZE];
  const size_t tidx = threadIdx.x;
  const size_t dx = (blockDim.x * gridDim.x) * elements_per_lane;
  const size_t gidx = (tidx + blockIdx.x * blockDim.x) * elements_per_lane;

  T tmp{0};
  for (size_t i = 0; i != chunks_per_block; ++i)
  {
    for (size_t j = 0; j != elements_per_lane; ++j)
    {
      tmp += load(a[gidx + i * dx + j]) * load(b[gidx + i * dx + j]);
    }
  }
  tb_sum[tidx] = tmp;
  __syncthreads();

  Reducer<>::reduce(tb_sum);
  if (tidx) return;
  store(tb_sum[0], sum[blockIdx.x]);
}

template <class T>
T HIPStream<T>::dot()
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(dot_kernel<elements_per_lane, chunks_per_block, T>),
                     dim3(block_count),
                     dim3(TBSIZE),
                     0, 0, d_a, d_b, d_sum);
  check_error();

  hipMemcpy(sums, d_sum, block_count*sizeof(T), hipMemcpyDeviceToHost);
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
