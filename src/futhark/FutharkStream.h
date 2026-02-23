// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
// Copyright (c) 2022 Troels Henriksen
// University of Copenhagen
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"
#include "babelstream.h"

#if defined(FUTHARK_BACKEND_c)
#define IMPLEMENTATION_STRING "Futhark (sequential)"
#elif defined(FUTHARK_BACKEND_multicore)
#define IMPLEMENTATION_STRING "Futhark (parallel CPU)"
#elif defined(FUTHARK_BACKEND_opencl)
#define IMPLEMENTATION_STRING "Futhark (OpencL)"
#elif defined(FUTHARK_BACKEND_cuda)
#define IMPLEMENTATION_STRING "Futhark (CUDA)"
#else
#define IMPLEMENTATION_STRING "Futhark (unknown backend)"
#endif

template <class T>
class FutharkStream : public Stream<T>
{
protected:
  // Size of arrays
  int array_size;
  // For device selection.
  std::string device;

  // Futhark stuff
  struct futhark_context_config *cfg;
  struct futhark_context *ctx;

  // Device side arrays
  void* a;
  void* b;
  void* c;

  // Host side arrays for verification
  std::vector<T> h_a, h_b, h_c;

public:
  FutharkStream(BenchId bs, const intptr_t array_size, const int device_id,
		T initA, T initB, T initC);
  ~FutharkStream();

  void copy() override;
  void add() override;
  void mul() override;
  void triad() override;
  void nstream() override;
  T dot() override;

  void get_arrays(T const*& a, T const*& b, T const*& c) override;  
  void init_arrays(T initA, T initB, T initC);
};
