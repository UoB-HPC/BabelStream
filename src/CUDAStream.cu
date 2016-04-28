
#include "CUDAStream.h"

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
CUDAStream<T>::CUDAStream(const unsigned int ARRAY_SIZE)
{
  array_size = ARRAY_SIZE;

  // Check buffers fit on the device
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  cudaMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
}


template <class T>
CUDAStream<T>::~CUDAStream()
{
  cudaFree(d_a);
  check_error();
  cudaFree(d_b);
  check_error();
  cudaFree(d_c);
  check_error();
}

template <class T>
void CUDAStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  // Copy host memory to device
  cudaMemcpy(d_a, a.data(), a.size()*sizeof(T), cudaMemcpyHostToDevice);
  check_error();
  cudaMemcpy(d_b, b.data(), b.size()*sizeof(T), cudaMemcpyHostToDevice);
  check_error();
  cudaMemcpy(d_c, c.data(), c.size()*sizeof(T), cudaMemcpyHostToDevice);
  check_error();
}

template <class T>
void CUDAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
  cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <class T>
void CUDAStream<T>::copy()
{
  copy_kernel<<<array_size/1024, 1024>>>(d_a, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = 3.0;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];
}

template <class T>
void CUDAStream<T>::mul()
{
  mul_kernel<<<array_size/1024, 1024>>>(d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <class T>
void CUDAStream<T>::add()
{
  add_kernel<<<array_size/1024, 1024>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = 3.0;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i] + scalar * c[i];
}

template <class T>
void CUDAStream<T>::triad()
{
  triad_kernel<<<array_size/1024, 1024>>>(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}


template <class T>
void CUDAStream<T>::listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cout << "No devices found." << std::endl;
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

template <class T>
std::string CUDAStream<T>::getDeviceName(const int device)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}

template <class T>
std::string CUDAStream<T>::getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error();
  int driver;
  cudaDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class CUDAStream<float>;
template class CUDAStream<double>;

