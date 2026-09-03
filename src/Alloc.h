// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

// Host allocation defaults, shared by every model that allocates its arrays on
// the host, so they all use the same alignment.

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
T *alloc_raw(size_t size)
{
  // aligned_alloc requires the size to be a multiple of the alignment, so round
  // it up. glibc lets the unrounded size through, but other libcs return a null
  // pointer, which used to show up as a crash for array sizes that are not a
  // multiple of ALIGNMENT.
  size_t bytes = sizeof(T) * size;
  size_t rounded = ((bytes + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

  T *ptr = (T *)aligned_alloc(ALIGNMENT, rounded);
  if (!ptr)
    throw std::runtime_error("Failed to allocate " + std::to_string(rounded) + " bytes");

  return ptr;
}

template <class T>
void dealloc_raw(T *ptr) { free(ptr); }
