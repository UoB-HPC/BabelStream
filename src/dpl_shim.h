#pragma once

#include <cstddef>

#ifdef USE_ONEDPL

// oneDPL C++17 PSTL

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>

#if ONEDPL_USE_DPCPP_BACKEND

#include <CL/sycl.hpp>

const static auto exe_policy = oneapi::dpl::execution::device_policy<>{
        oneapi::dpl::execution::make_device_policy(cl::sycl::default_selector{})
};

template<typename T>
T *alloc_raw(size_t size) { return sycl::malloc_shared<T>(size, exe_policy.queue()); }

template<typename T>
void dealloc_raw(T *ptr) { sycl::free(ptr, exe_policy.queue()); }

#define WORKAROUND

#else

// auto exe_policy = dpl::execution::seq;
// auto exe_policy = dpl::execution::par;
static constexpr auto exe_policy = dpl::execution::par_unseq;
#define USE_STD_PTR_ALLOC_DEALLOC
#define WORKAROUND

#endif

#else

// Normal C++17 PSTL

#include <algorithm>
#include <execution>
#include <numeric>

// auto exe_policy = std::execution::seq;
// auto exe_policy = std::execution::par;
static constexpr auto exe_policy = std::execution::par_unseq;
#define USE_STD_PTR_ALLOC_DEALLOC


#endif

#ifdef USE_STD_PTR_ALLOC_DEALLOC
#include "Alloc.h"
#endif

#ifdef USE_ONEDPL
#include <iostream>
inline void print_dpl_backend() {
  std::cout << "Using oneDPL backend: ";
#if ONEDPL_USE_DPCPP_BACKEND
  std::cout << "SYCL USM (device=" << exe_policy.queue().get_device().get_info<sycl::info::device::name>() << ")";
#elif ONEDPL_USE_TBB_BACKEND
  std::cout << "TBB " TBB_VERSION_STRING;
#elif ONEDPL_USE_OPENMP_BACKEND
  std::cout << "OpenMP";
#else
  std::cout << "Default";
#endif
  std::cout << std::endl;
}
#endif
