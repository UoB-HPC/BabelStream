
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "CUDAStream.h"

[[noreturn]] inline void error(char const* file, int line, char const* expr, cudaError_t e) {
  std::fprintf(stderr, "Error at %s:%d: %s (%d)\n  %s\n", file, line, cudaGetErrorString(e), e, expr);
  exit(e);
}

#define CU(EXPR) if (auto __e = (EXPR); __e != cudaSuccess) error(__FILE__, __LINE__, #EXPR, __e);

__host__ __device__ constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1)/b; }

template <class T>
CUDAStream<T>::CUDAStream(const int ARRAY_SIZE, const int device_index)
{
  // Set device
  int count;
  CU(cudaGetDeviceCount(&count));
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  CU(cudaSetDevice(device_index));

  // Print out device information
  std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
#if defined(MANAGED)
  std::cout << "Memory: MANAGED" << std::endl;
#elif defined(PAGEFAULT)
  std::cout << "Memory: PAGEFAULT" << std::endl;
#else
  std::cout << "Memory: DEFAULT" << std::endl;
#endif
  array_size = ARRAY_SIZE;


  // Query device for sensible dot kernel block count
  cudaDeviceProp props;
  CU(cudaGetDeviceProperties(&props, device_index));
  dot_num_blocks = props.multiProcessorCount * 4;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(sizeof(T) * dot_num_blocks);

  size_t array_bytes = sizeof(T);
  array_bytes *= ARRAY_SIZE;
  size_t total_bytes = array_bytes * 4;
  std::cout << "Reduction kernel config: " << dot_num_blocks << " groups of (fixed) size " << TBSIZE << std::endl;

  // Check buffers fit on the device
  if (props.totalGlobalMem < total_bytes)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
#if defined(MANAGED)
  CU(cudaMallocManaged(&d_a, array_bytes));
  CU(cudaMallocManaged(&d_b, array_bytes));
  CU(cudaMallocManaged(&d_c, array_bytes));
  CU(cudaMallocManaged(&d_sum, dot_num_blocks*sizeof(T)));
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(array_bytes);
  d_b = (T*)malloc(array_bytes);
  d_c = (T*)malloc(array_bytes);
  d_sum = (T*)malloc(sizeof(T)*dot_num_blocks);
#else
  CU(cudaMalloc(&d_a, array_bytes));
  CU(cudaMalloc(&d_b, array_bytes));
  CU(cudaMalloc(&d_c, array_bytes));
  CU(cudaMalloc(&d_sum, dot_num_blocks*sizeof(T)));
#endif
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  free(sums);

#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
  free(d_sum);
#else
  CU(cudaFree(d_a));
  CU(cudaFree(d_b));
  CU(cudaFree(d_c));
  CU(cudaFree(d_sum));
#endif
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC, int array_size)
{  
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < array_size; i += gridDim.x * blockDim.x) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
}

template <class T>
void CUDAStream<T>::init_arrays(T initA, T initB, T initC)
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  init_kernel<<<blocks, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaDeviceSynchronize());
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  CU(cudaDeviceSynchronize());
  for (int i = 0; i < array_size; i++)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  CU(cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost));
#endif
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c, int array_size)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < array_size; i += gridDim.x * blockDim.x) {
    c[i] = a[i];
  }
}

template <class T>
void CUDAStream<T>::copy()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  copy_kernel<<<blocks, TBSIZE>>>(d_a, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaDeviceSynchronize());
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c, int array_size)
{
  const T scalar = startScalar;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < array_size; i += gridDim.x * blockDim.x) {
    b[i] = scalar * c[i];
  }
}

template <class T>
void CUDAStream<T>::mul()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  mul_kernel<<<blocks, TBSIZE>>>(d_b, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaDeviceSynchronize());
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c, int array_size)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < array_size; i += gridDim.x * blockDim.x) {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void CUDAStream<T>::add()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  add_kernel<<<blocks, TBSIZE>>>(d_a, d_b, d_c, array_size);
  CU(cudaPeekAtLastError());  
  CU(cudaDeviceSynchronize());
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c, int array_size)
{
  const T scalar = startScalar;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < array_size; i += gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void CUDAStream<T>::triad()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  triad_kernel<<<blocks, TBSIZE>>>(d_a, d_b, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaDeviceSynchronize());
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c, int array_size)
{
  const T scalar = startScalar;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < array_size; i += gridDim.x * blockDim.x) {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
void CUDAStream<T>::nstream()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  nstream_kernel<<<blocks, TBSIZE>>>(d_a, d_b, d_c, array_size);
  CU(cudaPeekAtLastError());
  CU(cudaDeviceSynchronize());
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t local_i = threadIdx.x;

  tb_sum[local_i] = {};
  for (; i < array_size; i += blockDim.x*gridDim.x)
    tb_sum[local_i] += a[i] * b[i];

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
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
T CUDAStream<T>::dot()
{
  dot_kernel<<<dot_num_blocks, TBSIZE>>>(d_a, d_b, d_sum, array_size);
  CU(cudaPeekAtLastError());

#if defined(MANAGED) || defined(PAGEFAULT)
  CU(cudaDeviceSynchronize());
#else
  CU(cudaMemcpy(sums, d_sum, dot_num_blocks*sizeof(T), cudaMemcpyDeviceToHost));
#endif

  T sum = 0.0;
  for (int i = 0; i < dot_num_blocks; i++)
  {
#if defined(MANAGED) || defined(PAGEFAULT)
    sum += d_sum[i];
#else
    sum += sums[i];
#endif
  }

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
