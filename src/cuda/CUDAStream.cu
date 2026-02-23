// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "CUDAStream.h"
#include <nvml.h>

#if !defined(UNROLL_FACTOR)
#define UNROLL_FACTOR 4
#endif

[[noreturn]] inline void cuda_error(char const* file, int line, char const* expr, cudaError_t e) {
  std::fprintf(stderr, "CUDA Error at %s:%d: %s (%d)\n  %s\n", file, line, cudaGetErrorString(e), e, expr);
  exit(e);
}

[[noreturn]] inline void nvml_error(char const* file, int line, char const* expr, nvmlReturn_t e) {
  std::fprintf(stderr, "NVML Error at %s:%d: %s (%d)\n  %s\n", file, line, nvmlErrorString(e), e, expr);
  exit(e);
}

// The do while is there to make sure you remember to put a semi-colon after calling CU
#define CU(EXPR) do { auto __e = (EXPR); if (__e != cudaSuccess) cuda_error(__FILE__, __LINE__, #EXPR, __e); } while(false)
#define NVML(EXPR) do { auto __e = (EXPR); if (__e != NVML_SUCCESS) nvml_error(__FILE__, __LINE__, #EXPR, __e); } while(false)

// It is best practice to include __device__ and constexpr even though in BabelStream it only needs to be __host__ const
__host__ __device__ constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

cudaStream_t stream;

template <typename T>
T* alloc_device(const intptr_t array_size) {
  size_t array_bytes = sizeof(T) * array_size;
  T* p = nullptr;
#if defined(MANAGED)
  CU(cudaMallocManaged(&p, array_bytes));
#elif defined(PAGEFAULT)
  p = (T*)malloc(array_bytes);
#else
  CU(cudaMalloc(&p, array_bytes));
#endif
  if (p == nullptr) throw std::runtime_error("Failed to allocate device array");
  return p;
}

template <typename T>
T* alloc_host(const intptr_t array_size) {
  size_t array_bytes = sizeof(T) * array_size;
  T* p = nullptr;
#if defined(PAGEFAULT)
  p = (T*)malloc(array_bytes);
#else
  CU(cudaHostAlloc(&p, array_bytes, cudaHostAllocDefault));
#endif
  if (p == nullptr) throw std::runtime_error("Failed to allocate host array");
  return p;
}

template <typename T>
void free_device(T* p) {
#if defined(PAGEFAULT)
  free(p);
#else
  CU(cudaFree(p));
#endif
}

template <typename T>
void free_host(T* p) {
#if defined(PAGEFAULT)
  free(p);
#else
  CU(cudaFreeHost(p));
#endif
}

template <class T>
CUDAStream<T>::CUDAStream(BenchId bs, const intptr_t array_size, const int device_index,
			  T initA, T initB, T initC)
  : array_size(array_size)
{
  // Set device
  int count;
  CU(cudaGetDeviceCount(&count));
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  CU(cudaSetDevice(device_index));

  CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Print out device information
  std::cout << "CUDA Driver: " << getDeviceDriver(device_index) << std::endl;
  NVML(nvmlInit());
  cudaDeviceProp dprop;
  CU(cudaGetDeviceProperties(&dprop, device_index));
  unsigned int memclock;
  char mybus[16];
  sprintf(&mybus[0], "%04x:%02x:%02x.0", dprop.pciDomainID, dprop.pciBusID, dprop.pciDeviceID);
  nvmlDevice_t nvmldev;
  NVML(nvmlDeviceGetHandleByPciBusId(mybus, &nvmldev));
  NVML(nvmlDeviceGetClockInfo(nvmldev, NVML_CLOCK_MEM, &memclock));
  std::cout << "CUDA Device " << device_index << ": \""
	    << getDeviceName(device_index)
	    << "\" " << dprop.multiProcessorCount << " SMs(" << dprop.major << "," << dprop.minor << ") "
	    << "Memory: " << memclock << " MHz x " << dprop.memoryBusWidth << "-bit = "
	    << 2.0*memclock*(dprop.memoryBusWidth/8)/1000.0 << " GB/s PEAK, ECC is "
	    << (dprop.ECCEnabled ? "ON" : "OFF")
	    << std::endl;

  // Print Memory allocation API used for buffers
  std::cout << "Memory Allocation: ";
  #if defined(MANAGED)
      std::cout << "MANAGED";
  #elif defined(PAGEFAULT)
      std::cout << "PAGEFAULT";
  #else
      std::cout << "DEFAULT";
  #endif
  std::cout << std::endl;

  std::cout << "Parallel for kernel config: thread blocks of size " << TBSIZE << std::endl;

  // Set sensible dot kernel block count
  dot_num_blocks = dprop.multiProcessorCount * 4;

  // Size of partial sums for dot kernels
  size_t sums_bytes = sizeof(T) * dot_num_blocks;
  size_t array_bytes = sizeof(T) * array_size;
  size_t total_bytes = array_bytes * size_t(3) + sums_bytes;
  std::cout << "Reduction kernel config: " << dot_num_blocks << " groups of (fixed) size " << TBSIZE_DOT << std::endl;

  // Check buffers fit on the device
  if (dprop.totalGlobalMem < total_bytes) {
    std::cerr << "Requested array size of " << total_bytes * 1e-9
	      << " GB exceeds memory capacity of " << dprop.totalGlobalMem * 1e-9 << " GB !" << std::endl;
    throw std::runtime_error("Device does not have enough memory for all buffers");
  }

  // Allocate buffers:
  d_a = alloc_device<T>(array_size);
  d_b = alloc_device<T>(array_size);
  d_c = alloc_device<T>(array_size);
  sums = alloc_host<T>(dot_num_blocks);

  // Initialize buffers:
  init_arrays(initA, initB, initC);
}

template <class T>
CUDAStream<T>::~CUDAStream()
{
  CU(cudaStreamDestroy(stream));
  free_device(d_a);
  free_device(d_b);
  free_device(d_c);
  free_host(sums);
}

template <typename F>
__global__ void for_each_kernel(size_t array_size, size_t start, F f) {
  constexpr int unroll_factor = UNROLL_FACTOR;
#if defined(GRID_STRIDE)
  // Grid-stride loop
  size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x;
  #pragma unroll(unroll_factor)
  for (; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    f(i);
  }
#elif defined(BLOCK_STRIDE)
  // Block-stride loop
  size_t i = start * blockIdx.x + threadIdx.x;
  const size_t e = min(array_size, start * (blockIdx.x + size_t(1)) + threadIdx.x);
  #pragma unroll(unroll_factor)
  for (; i < e; i += blockDim.x) {
    f(i);
  }
#else
  #error Must pick grid-stride or block-stride loop
#endif
}

template <typename F>
void for_each(size_t array_size, F f) {
  static int threads_per_block = 0;
  if (threads_per_block == 0) {
    // Pick suitable thread block size for F:
    int min_blocks_per_grid;
    auto dyn_smem = [] __host__ __device__ (int){ return 0; };
    CU(cudaOccupancyMaxPotentialBlockSizeVariableSMem
       (&min_blocks_per_grid, &threads_per_block, for_each_kernel<F>, dyn_smem, 0));
    // Clamp to TBSIZE
    threads_per_block = std::min(TBSIZE, threads_per_block);
  }
  size_t blocks = ceil_div(array_size / UNROLL_FACTOR, threads_per_block);
  size_t start = ceil_div(array_size, (size_t)blocks);
  for_each_kernel<<<blocks, TBSIZE, 0, stream>>>(array_size, start, f);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  for_each(array_size, [=,a=d_a,b=d_b,c=d_c] __device__ (size_t i) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  });
}

template <class T>
void CUDAStream<T>::get_arrays(T const*& a, T const*& b, T const*& c)
{
  CU(cudaStreamSynchronize(stream));
#if defined(PAGEFAULT) || defined(MANAGED)
  // Unified memory: return pointers to device memory
  a = d_a;
  b = d_b;
  c = d_c;
#else
  // No Unified memory: copy data to the host
  size_t nbytes = array_size * sizeof(T);
  h_a.resize(array_size);
  h_b.resize(array_size);
  h_c.resize(array_size);
  a = h_a.data();
  b = h_b.data();
  c = h_c.data();
  CU(cudaMemcpy(h_a.data(), d_a, nbytes, cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(h_b.data(), d_b, nbytes, cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(h_c.data(), d_c, nbytes, cudaMemcpyDeviceToHost));
#endif
}

template <class T>
void CUDAStream<T>::copy()
{
  for_each(array_size, [a=d_a,c=d_c] __device__ (size_t i) {
    c[i] = a[i];
  });
}

template <class T>
void CUDAStream<T>::mul()
{
  for_each(array_size, [b=d_b,c=d_c] __device__ (size_t i) {
    b[i] = startScalar * c[i];
  });
}

template <class T>
void CUDAStream<T>::add()
{
  for_each(array_size, [a=d_a,b=d_b,c=d_c] __device__ (size_t i) {
    c[i] = a[i] + b[i];
  });
}

template <class T>
void CUDAStream<T>::triad()
{
  for_each(array_size, [a=d_a,b=d_b,c=d_c] __device__ (size_t i) {
    a[i] = b[i] + startScalar * c[i];
  });
}

template <class T>
void CUDAStream<T>::nstream()
{
  for_each(array_size, [a=d_a,b=d_b,c=d_c] __device__ (size_t i) {
    a[i] += b[i] + startScalar * c[i];
  });
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T* sums, size_t array_size)
{
  __shared__ T smem[TBSIZE_DOT];
  T tmp = T(0.);
  const size_t tidx = threadIdx.x;
  size_t i = tidx + (size_t)blockDim.x * blockIdx.x;
  for (; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    tmp += a[i] * b[i];
  }
  smem[tidx] = tmp;

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    __syncthreads();
    if (tidx < offset) smem[tidx] += smem[tidx+offset];
  }

  // First thread writes to host memory directly from the device
  if (tidx == 0) sums[blockIdx.x] = smem[tidx];
}

template <class T>
T CUDAStream<T>::dot()
{
  dot_kernel<<<dot_num_blocks, TBSIZE_DOT, 0, stream>>>(d_a, d_b, sums, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaStreamSynchronize(stream));

  T sum = 0.0;
  for (intptr_t i = 0; i < dot_num_blocks; ++i) sum += sums[i];

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  CU(cudaGetDeviceCount(&count));

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
  cudaDeviceProp props;
  CU(cudaGetDeviceProperties(&props, device));
  return std::string(props.name);
}

std::string getDeviceDriver(const int device)
{
  CU(cudaSetDevice(device));
  int driver;
  CU(cudaDriverGetVersion(&driver));
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;
