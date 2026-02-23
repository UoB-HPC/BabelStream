
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include "benchmark.h"

#ifdef ENABLE_CALIPER
#include <caliper/cali.h>
#include <adiak.h>
#endif

using std::intptr_t;

// Array values
#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

template <class T>
class Stream
{
  public:
    virtual ~Stream(){}

    // Kernels
    // These must be blocking calls
    virtual void copy() = 0;
    virtual void mul() = 0;
    virtual void add() = 0;
    virtual void triad() = 0;
    virtual void nstream() = 0;
    virtual T dot() = 0;

    // Set pointers to read from arrays
    virtual void get_arrays(T const*& a, T const*& b, T const*& c) = 0;
};

// Implementation specific device functions
void listDevices(void);
std::string getDeviceName(const int);
std::string getDeviceDriver(const int);

