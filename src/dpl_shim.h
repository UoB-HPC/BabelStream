#pragma once

#include <cstdlib>
#include <cstddef>

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

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

#else

// auto exe_policy = dpl::execution::seq;
// auto exe_policy = dpl::execution::par;
static constexpr auto exe_policy = dpl::execution::par_unseq;
#define USE_STD_PTR_ALLOC_DEALLOC

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

#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
#include <CL/sycl.hpp>

// TODO We temporarily use malloc_shared/free here for hipSYCL stdpar because there's a linking issue if we let it hijack new/delete
//  for this to work, we compile with --hipsycl-stdpar-system-usm so that hijacking is disabled
static cl::sycl::queue queue{cl::sycl::default_selector_v};
template <typename T> T *alloc_raw(size_t size) { return cl::sycl::malloc_shared<T>(size, queue); }
template <typename T> void dealloc_raw(T *ptr) { cl::sycl::free(ptr, queue); }

#else
template<typename T>
T *alloc_raw(size_t size) { return (T *) aligned_alloc(ALIGNMENT, sizeof(T) * size); }

template<typename T>
void dealloc_raw(T *ptr) { free(ptr); }
#endif

#endif
