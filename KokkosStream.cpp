// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "KokkosStream.hpp"

template <class T>
KokkosStream<T>::KokkosStream(
        const int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE)
{
  Kokkos::initialize();

  d_a = new Kokkos::View<T*>("d_a", ARRAY_SIZE);
  d_b = new Kokkos::View<T*>("d_b", ARRAY_SIZE);
  d_c = new Kokkos::View<T*>("d_c", ARRAY_SIZE);
  hm_a = new typename Kokkos::View<T*>::HostMirror();
  hm_b = new typename Kokkos::View<T*>::HostMirror();
  hm_c = new typename Kokkos::View<T*>::HostMirror();
  *hm_a = create_mirror_view(*d_a);
  *hm_b = create_mirror_view(*d_b);
  *hm_c = create_mirror_view(*d_c);
}

template <class T>
KokkosStream<T>::~KokkosStream()
{
  Kokkos::finalize();
}

template <class T>
void KokkosStream<T>::init_arrays(T initA, T initB, T initC)
{
  Kokkos::View<T*> a(*d_a);
  Kokkos::View<T*> b(*d_b);
  Kokkos::View<T*> c(*d_c);
  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    a[index] = initA;
    b[index] = initB;
    c[index] = initC;
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::read_arrays(
        std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  deep_copy(*hm_a, *d_a);
  deep_copy(*hm_b, *d_b);
  deep_copy(*hm_c, *d_c);
  for(int ii = 0; ii < array_size; ++ii)
  {
    a[ii] = (*hm_a)(ii);
    b[ii] = (*hm_b)(ii);
    c[ii] = (*hm_c)(ii);
  }
}

template <class T>
void KokkosStream<T>::copy()
{
  Kokkos::View<T*> a(*d_a);
  Kokkos::View<T*> b(*d_b);
  Kokkos::View<T*> c(*d_c);

  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    c[index] = a[index];
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::mul()
{
  Kokkos::View<T*> a(*d_a);
  Kokkos::View<T*> b(*d_b);
  Kokkos::View<T*> c(*d_c);

  const T scalar = startScalar;
  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    b[index] = scalar*c[index];
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::add()
{
  Kokkos::View<T*> a(*d_a);
  Kokkos::View<T*> b(*d_b);
  Kokkos::View<T*> c(*d_c);

  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    c[index] = a[index] + b[index];
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::triad()
{
  Kokkos::View<T*> a(*d_a);
  Kokkos::View<T*> b(*d_b);
  Kokkos::View<T*> c(*d_c);

  const T scalar = startScalar;
  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    a[index] = b[index] + scalar*c[index];
  });
  Kokkos::fence();
}

template <class T>
void KokkosStream<T>::nstream()
{
  Kokkos::View<T*> a(*d_a);
  Kokkos::View<T*> b(*d_b);
  Kokkos::View<T*> c(*d_c);

  const T scalar = startScalar;
  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    a[index] += b[index] + scalar*c[index];
  });
  Kokkos::fence();
}

template <class T>
T KokkosStream<T>::dot()
{
  Kokkos::View<T*> a(*d_a);
  Kokkos::View<T*> b(*d_b);

  T sum = 0.0;

  Kokkos::parallel_reduce(array_size, KOKKOS_LAMBDA (const long index, T &tmp)
  {
    tmp += a[index] * b[index];
  }, sum);

  return sum;

}

void listDevices(void)
{
  std::cout << "Kokkos library for " << getDeviceName(0) << std::endl;
}


std::string getDeviceName(const int device)
{
  return typeid (Kokkos::DefaultExecutionSpace).name();
}


std::string getDeviceDriver(const int device)
{
  return "Kokkos";
}

template class KokkosStream<float>;
template class KokkosStream<double>;
