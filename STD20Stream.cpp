// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "STD20Stream.hpp"

#include <algorithm>
#include <execution>
#include <ranges>

template <class T>
STD20Stream<T>::STD20Stream(const int ARRAY_SIZE, int device)
 : array_size{ARRAY_SIZE}
{
  a = std::vector<T>(array_size);
  b = std::vector<T>(array_size);
  c = std::vector<T>(array_size);
}

template <class T>
void STD20Stream<T>::init_arrays(T initA, T initB, T initC)
{
  std::for_each_n(
    std::execution::par_unseq,
    std::views::iota(0).begin(), array_size, // loop range
    [&] (int i) {
      a[i] = initA;
      b[i] = initB;
      c[i] = initC;
    }
  );
}

template <class T>
void STD20Stream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  // Element-wise copy.
  h_a = a;
  h_b = b;
  h_c = c;
}

template <class T>
void STD20Stream<T>::copy()
{
  std::for_each_n(
    std::execution::par_unseq,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      c[i] = a[i];
    }
  );
}

template <class T>
void STD20Stream<T>::mul()
{
  const T scalar = startScalar;

  std::for_each_n(
    std::execution::par_unseq,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      b[i] = scalar * c[i];
    }
  );
}

template <class T>
void STD20Stream<T>::add()
{
  std::for_each_n(
    std::execution::par_unseq,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      c[i] = a[i] + b[i];
    }
  );
}

template <class T>
void STD20Stream<T>::triad()
{
  const T scalar = startScalar;

  std::for_each_n(
    std::execution::par_unseq,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      a[i] = b[i] + scalar * c[i];
    }
  );
}

template <class T>
void STD20Stream<T>::nstream()
{
  const T scalar = startScalar;

  std::for_each_n(
    std::execution::par_unseq,
    std::views::iota(0).begin(), array_size,
    [&] (int i) {
      a[i] += b[i] + scalar * c[i];
    }
  );
}

template <class T>
T STD20Stream<T>::dot()
{
  // sum += a[i] * b[i];
  return
    std::transform_reduce(
      std::execution::par_unseq,
      a.begin(), a.end(), b.begin(), 0.0);
}

void listDevices(void)
{
  std::cout << "C++20 does not expose devices" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}

template class STD20Stream<float>;
template class STD20Stream<double>;

