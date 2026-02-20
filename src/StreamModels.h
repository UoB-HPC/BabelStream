#pragma once
#include <memory>

#if defined(CUDA)
#include "CUDAStream.h"
#elif defined(STD)
#include "STDStream.h"
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
#elif defined(SERIAL)
#include "SerialStream.h"
#elif defined(FUTHARK)
#include "FutharkStream.h"
#endif

template <typename T, typename...Args>
std::unique_ptr<Stream<T>> make_stream(Args... args) {
#if defined(CUDA)
  // Use the CUDA implementation
  return std::make_unique<CUDAStream<T>>(args...);

#elif defined(HIP)
  // Use the HIP implementation
  return std::make_unique<HIPStream<T>>(args...);

#elif defined(HC)
  // Use the HC implementation
  return std::make_unique<HCStream<T>>(args...);

#elif defined(OCL)
  // Use the OpenCL implementation
  return std::make_unique<OCLStream<T>>(args...);

#elif defined(USE_RAJA)
  // Use the RAJA implementation
  return std::make_unique<RAJAStream<T>>(args...);

#elif defined(KOKKOS)
  // Use the Kokkos implementation
  return std::make_unique<KokkosStream<T>>(args...);

#elif defined(STD)
  // Use the C++ STD data-oriented implementation
  return std::make_unique<STDStream<T>>(args...);

#elif defined(TBB)
  // Use the C++20 implementation
  return std::make_unique<TBBStream<T>>(args...);

#elif defined(THRUST)
  // Use the Thrust implementation
  return std::make_unique<ThrustStream<T>>(args...);

#elif defined(ACC)
  // Use the OpenACC implementation
  return std::make_unique<ACCStream<T>>(args...);

#elif defined(SYCL) || defined(SYCL2020)
  // Use the SYCL implementation
  return std::make_unique<SYCLStream<T>>(args...);

#elif defined(OMP)
  // Use the OpenMP implementation
  return std::make_unique<OMPStream<T>>(args...);

#elif defined(SERIAL)
  // Use the Serial implementation
  return std::make_unique<SerialStream<T>>(args...);

#elif defined(FUTHARK)
  // Use the Futhark implementation
  return std::make_unique<FutharkStream<T>>(args...);

#else

  #error unknown benchmark

#endif
}
