
#include "MUSAStream.h"

void check_error(void)
{
  musaError_t err = musaGetLastError();
  if (err != musaSuccess)
  {
    std::cerr << "Error: " << musaGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
MUSAStream<T>::MUSAStream(const int ARRAY_SIZE, const int device_index)
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
  musaGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  musaSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using MUSA device " << getDeviceName(device_index) << std::endl;
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
  musaDeviceProp props;
  musaGetDeviceProperties(&props, device_index);
  check_error();
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
  // Use managed memory on MUSA devices may not work as expected
  musaMallocManaged(&d_a, array_bytes);
  check_error();
  musaMallocManaged(&d_b, array_bytes);
  check_error();
  musaMallocManaged(&d_c, array_bytes);
  check_error();
  musaMallocManaged(&d_sum, dot_num_blocks*sizeof(T));
  check_error();
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(array_bytes);
  d_b = (T*)malloc(array_bytes);
  d_c = (T*)malloc(array_bytes);
  d_sum = (T*)malloc(sizeof(T)*dot_num_blocks);
#else
  musaMalloc(&d_a, array_bytes);
  check_error();
  musaMalloc(&d_b, array_bytes);
  check_error();
  musaMalloc(&d_c, array_bytes);
  check_error();
  musaMalloc(&d_sum, dot_num_blocks*sizeof(T));
  check_error();
#endif
}


template <class T>
MUSAStream<T>::~MUSAStream()
{
  free(sums);

#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
  free(d_sum);
#else
  musaFree(d_a);
  check_error();
  musaFree(d_b);
  check_error();
  musaFree(d_c);
  check_error();
  musaFree(d_sum);
  check_error();
#endif
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void MUSAStream<T>::init_arrays(T initA, T initB, T initC)
{
  init_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  musaDeviceSynchronize();
  check_error();
}

template <class T>
void MUSAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  musaDeviceSynchronize();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  musaMemcpy(a.data(), d_a, a.size()*sizeof(T), musaMemcpyDeviceToHost);
  check_error();
  musaMemcpy(b.data(), d_b, b.size()*sizeof(T), musaMemcpyDeviceToHost);
  check_error();
  musaMemcpy(c.data(), d_c, c.size()*sizeof(T), musaMemcpyDeviceToHost);
  check_error();
#endif
}


template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <class T>
void MUSAStream<T>::copy()
{
  copy_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_c);
  check_error();
  musaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  b[i] = scalar * c[i];
}

template <class T>
void MUSAStream<T>::mul()
{
  mul_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_b, d_c);
  check_error();
  musaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <class T>
void MUSAStream<T>::add()
{
  add_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  musaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = b[i] + scalar * c[i];
}

template <class T>
void MUSAStream<T>::triad()
{
  triad_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  musaDeviceSynchronize();
  check_error();
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] += b[i] + scalar * c[i];
}

template <class T>
void MUSAStream<T>::nstream()
{
  nstream_kernel<<<array_size/TBSIZE, TBSIZE>>>(d_a, d_b, d_c);
  check_error();
  musaDeviceSynchronize();
  check_error();
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
T MUSAStream<T>::dot()
{
  dot_kernel<<<dot_num_blocks, TBSIZE>>>(d_a, d_b, d_sum, array_size);
  check_error();

#if defined(MANAGED) || defined(PAGEFAULT)
  musaDeviceSynchronize();
  check_error();
#else
  musaMemcpy(sums, d_sum, dot_num_blocks*sizeof(T), musaMemcpyDeviceToHost);
  check_error();
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
  musaGetDeviceCount(&count);
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
  musaDeviceProp props;
  musaGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  musaSetDevice(device);
  check_error();
  int driver;
  musaDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class MUSAStream<float>;
template class MUSAStream<double>;
