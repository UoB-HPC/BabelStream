
#include "OCLStream.h"

std::string kernels{R"CLC(

  constant TYPE scalar = 3.0;

  kernel void copy(
    global const TYPE * restrict a,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i];
  }

  kernel void mul(
    global TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    b[i] = scalar * c[i];
  }

  kernel void add(
    global const TYPE * restrict a,
    global const TYPE * restrict b,
    global TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    c[i] = a[i] + b[i];
  }

  kernel void triad(
    global TYPE * restrict a,
    global const TYPE * restrict b,
    global const TYPE * restrict c)
  {
    const size_t i = get_global_id(0);
    a[i] = b[i] + scalar * c[i];
  }

)CLC"};


template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE)
{
  array_size = ARRAY_SIZE;

  // Setup default OpenCL GPU
  context = cl::Context::getDefault();
  queue = cl::CommandQueue::getDefault();

  // Create program
  cl::Program program(kernels);
  if (sizeof(T) == sizeof(double))
    program.build("-DTYPE=double");
  else if (sizeof(T) == sizeof(float))
    program.build("-DTYPE=float");

  // Create kernels
  copy_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "copy");
  mul_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "mul");
  add_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "add");
  triad_kernel = new cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "triad");

  // Create buffers
  d_a = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_b = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);
  d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(T) * ARRAY_SIZE);

}

template <class T>
OCLStream<T>::~OCLStream()
{
  delete[] copy_kernel;
  delete[] mul_kernel;
  delete[] add_kernel;
  delete[] triad_kernel;
}

template <class T>
void OCLStream<T>::copy()
{
  (*copy_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::mul()
{
  (*mul_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::add()
{
  (*add_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::triad()
{
  (*triad_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(array_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  
  return;
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  return;
}


template class OCLStream<float>;
template class OCLStream<double>;

