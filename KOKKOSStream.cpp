// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "KOKKOSStream.hpp"

using namespace Kokkos;

template <class T>
KOKKOSStream<T>::KOKKOSStream(
        const unsigned int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE)
{
  Kokkos::initialize();

  d_a = new View<double*, DEVICE>("d_a", ARRAY_SIZE);
  d_b = new View<double*, DEVICE>("d_b", ARRAY_SIZE);
  d_c = new View<double*, DEVICE>("d_c", ARRAY_SIZE);
  hm_a = new View<double*, DEVICE>::HostMirror();
  hm_b = new View<double*, DEVICE>::HostMirror();
  hm_c = new View<double*, DEVICE>::HostMirror();
  *hm_a = create_mirror_view(*d_a);
  *hm_b = create_mirror_view(*d_b);
  *hm_c = create_mirror_view(*d_c);
}

template <class T>
KOKKOSStream<T>::~KOKKOSStream()
{
  finalize();
}

template <class T>
void KOKKOSStream<T>::init_arrays(T initA, T initB, T initC)
{
  View<double*, DEVICE> a(*d_a);
  View<double*, DEVICE> b(*d_b);
  View<double*, DEVICE> c(*d_c);
  parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    a[index] = initA;
    b[index] = initB;
    c[index] = initC;
  });
  Kokkos::fence();
}

template <class T>
void KOKKOSStream<T>::read_arrays(
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
void KOKKOSStream<T>::copy()
{
  View<double*, DEVICE> a(*d_a);
  View<double*, DEVICE> b(*d_b);
  View<double*, DEVICE> c(*d_c);

  parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    c[index] = a[index];
  });
  Kokkos::fence();
}

template <class T>
void KOKKOSStream<T>::mul()
{
  View<double*, DEVICE> a(*d_a);
  View<double*, DEVICE> b(*d_b);
  View<double*, DEVICE> c(*d_c);

  const T scalar = startScalar;
  parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    b[index] = scalar*c[index];
  });
  Kokkos::fence();
}

template <class T>
void KOKKOSStream<T>::add()
{
  View<double*, DEVICE> a(*d_a);
  View<double*, DEVICE> b(*d_b);
  View<double*, DEVICE> c(*d_c);

  parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    c[index] = a[index] + b[index];
  });
  Kokkos::fence();
}

template <class T>
void KOKKOSStream<T>::triad()
{
  View<double*, DEVICE> a(*d_a);
  View<double*, DEVICE> b(*d_b);
  View<double*, DEVICE> c(*d_c);

  const T scalar = startScalar;
  parallel_for(array_size, KOKKOS_LAMBDA (const long index)
  {
    a[index] = b[index] + scalar*c[index];
  });
  Kokkos::fence();
}

template <class T>
T KOKKOSStream<T>::dot()
{
  View<double *, DEVICE> a(*d_a);
  View<double *, DEVICE> b(*d_b);

  T sum = 0.0;

  parallel_reduce(array_size, KOKKOS_LAMBDA (const long index, double &tmp)
  {
    tmp += a[index] * b[index];
  }, sum);

  return sum;

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

//template class KOKKOSStream<float>;
template class KOKKOSStream<double>;
