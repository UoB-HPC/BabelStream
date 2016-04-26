
#include "CUDAStream.h"

template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i];
}

template <class T>
void CUDAStream<T>::copy()
{
  copy_kernel<<<1024, 1024>>>(d_a, d_c);
}

template <class T>
void CUDAStream<T>::mul()
{
  return;
}

template <class T>
void CUDAStream<T>::add()
{
  return;
}

template <class T>
void CUDAStream<T>::triad()
{
  return;
}

template class CUDAStream<float>;
template class CUDAStream<double>;
