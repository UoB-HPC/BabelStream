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

public:
  FutharkStream(const int, int);
  ~FutharkStream();

  virtual void copy() override;
  virtual void add() override;
  virtual void mul() override;
  virtual void triad() override;
  virtual void nstream() override;
  virtual T dot() override;

  virtual void init_arrays(T initA, T initB, T initC) override;
  virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};
