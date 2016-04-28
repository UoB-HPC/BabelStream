
#include "OCLStream.h"

template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE)
{
  array_size = ARRAY_SIZE;

  // Setup default OpenCL GPU
  context = cl::Context::getDefault();
  //queue = cl::CommandQueue::getDefault();

  // Create program

  std::string kernels{R"CLC(

    const double scalar = 3.0;

    kernel void copy(
      global const double * restrict a,
      global double * restrict c)
    {
      const size_t i = get_global_id(0);
      c[i] = a[i];
    }
  )CLC"};
 
std::cout << kernels << std::endl;
 
  //cl::Program program(kernels);
  //program.build();

exit(-1);


}

template <class T>
void OCLStream<T>::copy()
{
  return;
}

template <class T>
void OCLStream<T>::mul()
{
  return;
}

template <class T>
void OCLStream<T>::add()
{
  return;
}

template <class T>
void OCLStream<T>::triad()
{
  return;
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

