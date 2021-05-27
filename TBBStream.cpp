// Copyright (c) 2020 Tom Deakin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "TBBStream.hpp"
#include "tbb/tbb.h"

template <class T>
TBBStream<T>::TBBStream(const int ARRAY_SIZE, int device)
 : partitioner(static_cast<Partitioner>(device)), range(0, ARRAY_SIZE), a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE)
{
  std::cout << "Using TBB partitioner: " << getDeviceName(device) << std::endl;
}

template <class T>
template <typename U, typename F>
U TBBStream<T>::with_partitioner(const F &f) 
{
  switch(partitioner){
    case Partitioner::Auto:      return f(tbb::auto_partitioner{});
    case Partitioner::Affinity:  { tbb::affinity_partitioner p; return f(p); }  //  parallel_* doesn't take const affinity_partitioner here
    case Partitioner::Static:    return f(tbb::static_partitioner{});
    case Partitioner::Simple:    return f(tbb::simple_partitioner{});
    default:                     throw std::runtime_error("Error asking for name for non-existant device");
  }
}

template <class T>
template <typename F>
void TBBStream<T>::parallel_for(const F &f) 
{
  // using size_t as per the range type (also used in the official documentation)
  with_partitioner<std::nullptr_t>([&](auto &&p) { 
    tbb::parallel_for(range, [&](const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) { 
        f(i);
      }
    }, p);
    return nullptr; // what we really want here is std::monostate, but we don't want to be C++17 only so nullptr_t it is
  });
}

template <class T>
template <typename F, typename Op>
T TBBStream<T>::parallel_reduce(T init, const Op &op, const F &f) 
{
  return with_partitioner<T>([&](auto &&p) {
    return tbb::parallel_reduce(range, init, [&](const tbb::blocked_range<size_t>& r, T acc) {
      for (size_t i = r.begin(); i < r.end(); ++i) { 
        acc = op(acc, f(i));
      }
      return acc;
    }, op, p);
  });
}

template <class T>
void TBBStream<T>::init_arrays(T initA, T initB, T initC)
{

  parallel_for([&](size_t i){ 
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;  
  });

}

template <class T>
void TBBStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  // Element-wise copy.
  h_a = a;
  h_b = b;
  h_c = c;
}

template <class T>
void TBBStream<T>::copy()
{
  parallel_for([&](size_t i){ c[i] = a[i]; });
}

template <class T>
void TBBStream<T>::mul()
{
  const T scalar = startScalar;
  
  parallel_for([&](size_t i){ b[i] = scalar * c[i]; });
  
}

template <class T>
void TBBStream<T>::add()
{

  parallel_for([&](size_t i){ c[i] = a[i] + b[i]; });

}

template <class T>
void TBBStream<T>::triad()
{
  const T scalar = startScalar;

  parallel_for([&](size_t i){ a[i] = b[i] + scalar * c[i]; });

}

template <class T>
void TBBStream<T>::nstream()
{
  const T scalar = startScalar;

  parallel_for([&](size_t i){ a[i] += b[i] + scalar * c[i]; });

}

template <class T>
T TBBStream<T>::dot()
{
  // sum += a[i] * b[i];
  return parallel_reduce(0.0, std::plus<T>(), [&](size_t i) { return a[i] * b[i]; });
}

void listDevices(void)
{
  std::cout 
    << "[" << static_cast<int>(Partitioner::Auto) << "] auto partitioner\n" 
    << "[" << static_cast<int>(Partitioner::Affinity) << "] affinity partitioner\n" 
    << "[" << static_cast<int>(Partitioner::Static) << "] static partitioner\n" 
    << "[" << static_cast<int>(Partitioner::Simple) << "] simple partitioner\n" 
    << "See https://spec.oneapi.com/versions/latest/elements/oneTBB/source/algorithms.html#partitioners for more details" 
    << std::endl;
}

std::string getDeviceName(const int device)
{
  switch(static_cast<Partitioner>(device)){
    case Partitioner::Auto:      return "auto_partitioner";
    case Partitioner::Affinity:  return "affinity_partitioner";
    case Partitioner::Static:    return "static_partitioner";
    case Partitioner::Simple:    return "simple_partitioner";
    default:                     throw std::runtime_error("Error asking for name for non-existant device");
  }
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}

template class TBBStream<float>;
template class TBBStream<double>;

