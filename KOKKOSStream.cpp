// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "KOKKOSStream.hpp"

using Kokkos::parallel_for;

template <class T>
KOKKOSStream<T>::KOKKOSStream(
        const unsigned int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE)
{
  Kokkos::initialize();

  new(d_a) Kokkos::View<double*, DEVICE>("d_a", ARRAY_SIZE);
  new(d_b) Kokkos::View<double*, DEVICE>("d_b", ARRAY_SIZE);
  new(d_c) Kokkos::View<double*, DEVICE>("d_c", ARRAY_SIZE);
  new(hm_a) Kokkos::View<double*>::HostMirror();
  new(hm_b) Kokkos::View<double*>::HostMirror();
  new(hm_c) Kokkos::View<double*>::HostMirror();
  hm_a = Kokkos::create_mirror_view(d_a);
  hm_b = Kokkos::create_mirror_view(d_b);
  hm_c = Kokkos::create_mirror_view(d_c);
}

template <class T>
KOKKOSStream<T>::~KOKKOSStream()
{
  Kokkos::finalize();
}

template <class T>
void KOKKOSStream<T>::write_arrays(
        const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  for(int ii = 0; ii < array_size; ++ii)
  {
    hm_a(ii) = a[ii];
    hm_b(ii) = b[ii];
    hm_c(ii) = c[ii];
  }
  Kokkos::deep_copy(hm_a, d_a);
  Kokkos::deep_copy(hm_b, d_b);
  Kokkos::deep_copy(hm_c, d_c);
}

template <class T>
void KOKKOSStream<T>::read_arrays(
        std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  Kokkos::deep_copy(d_a, hm_a);
  Kokkos::deep_copy(d_a, hm_b);
  Kokkos::deep_copy(d_a, hm_c);
  for(int ii = 0; ii < array_size; ++ii)
  {
    a[ii] = hm_a(ii);
    b[ii] = hm_b(ii);
    c[ii] = hm_c(ii);
  }
}

template <class T>
void KOKKOSStream<T>::copy()
{
  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const int index) 
  {
    d_c[index] = d_a[index];
  });
}

template <class T>
void KOKKOSStream<T>::mul()
{
  const T scalar = 3.0;
  parallel_for(array_size, KOKKOS_LAMBDA (const int index) 
  {
    d_b[index] = scalar*d_c[index];
  });
}

template <class T>
void KOKKOSStream<T>::add()
{
  parallel_for(array_size, KOKKOS_LAMBDA (const int index) 
  {
    d_c[index] = d_a[index] + d_b[index];
  });
}

template <class T>
void KOKKOSStream<T>::triad()
{
  const T scalar = 3.0;
  parallel_for(array_size, KOKKOS_LAMBDA (const int index) 
  {
    d_a[index] = d_b[index] + scalar*d_c[index];
  });
}

void listDevices(void)
{
  std::cout << "This is not the device you are looking for.";
}


std::string getDeviceName(const int device)
{
  return "Kokkos";
}


std::string getDeviceDriver(const int device)
{
  return "Kokkos";
}

template class KOKKOSStream<float>;
template class KOKKOSStream<double>;

