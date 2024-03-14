#pragma once
#include <memory>

#if defined(CUDA)
#include "CUDAStream.h"
#elif defined(STD_DATA)
#include "STDDataStream.h"
#elif defined(STD_INDICES)
#include "STDIndicesStream.h"
#elif defined(STD_RANGES)
#include "STDRangesStream.hpp"
#elif defined(TBB)
#include "TBBStream.hpp"
#elif defined(THRUST)
#include "ThrustStream.h"
#elif defined(HIP)
#include "HIPStream.h"
#elif defined(HC)
#include "HCStream.h"
#elif defined(OCL)
#include "OCLStream.h"
#elif defined(USE_RAJA)
#include "RAJAStream.hpp"
#elif defined(KOKKOS)
#include "KokkosStream.hpp"
#elif defined(ACC)
#include "ACCStream.h"
#elif defined(SYCL)
#include "SYCLStream.h"
#elif defined(SYCL2020)
#include "SYCLStream2020.h"
#elif defined(OMP)
#include "OMPStream.h"
#elif defined(FUTHARK)
#include "FutharkStream.h"
#endif

template <typename T>
std::unique_ptr<Stream<T>> make_stream(intptr_t array_size, int deviceIndex) {
#if defined(CUDA)
  // Use the CUDA implementation
  return std::make_unique<CUDAStream<T>>(array_size, deviceIndex);

#elif defined(HIP)
  // Use the HIP implementation
  return std::make_unique<HIPStream<T>>(array_size, deviceIndex);

#elif defined(HC)
  // Use the HC implementation
  return std::make_unique<HCStream<T>>(array_size, deviceIndex);

#elif defined(OCL)
  // Use the OpenCL implementation
  return std::make_unique<OCLStream<T>>(array_size, deviceIndex);

#elif defined(USE_RAJA)
  // Use the RAJA implementation
  return std::make_unique<RAJAStream<T>>(array_size, deviceIndex);

#elif defined(KOKKOS)
  // Use the Kokkos implementation
  return std::make_unique<KokkosStream<T>>(array_size, deviceIndex);

#elif defined(STD_DATA)
  // Use the C++ STD data-oriented implementation
  return std::make_unique<STDDataStream<T>>(array_size, deviceIndex);

#elif defined(STD_INDICES)
  // Use the C++ STD index-oriented implementation
  return std::make_unique<STDIndicesStream<T>>(array_size, deviceIndex);

#elif defined(STD_RANGES)
  // Use the C++ STD ranges implementation
  return std::make_unique<STDRangesStream<T>>(array_size, deviceIndex);

#elif defined(TBB)
  // Use the C++20 implementation
  return std::make_unique<TBBStream<T>>(array_size, deviceIndex);

#elif defined(THRUST)
  // Use the Thrust implementation
  return std::make_unique<ThrustStream<T>>(array_size, deviceIndex);

#elif defined(ACC)
  // Use the OpenACC implementation
  return std::make_unique<ACCStream<T>>(array_size, deviceIndex);

#elif defined(SYCL) || defined(SYCL2020)
  // Use the SYCL implementation
  return std::make_unique<SYCLStream<T>>(array_size, deviceIndex);

#elif defined(OMP)
  // Use the OpenMP implementation
  return std::make_unique<OMPStream<T>>(array_size, deviceIndex);

#elif defined(FUTHARK)
  // Use the Futhark implementation
  return std::make_unique<FutharkStream<T>>(array_size, deviceIndex);

#else

  #error unknown benchmark

#endif
}
