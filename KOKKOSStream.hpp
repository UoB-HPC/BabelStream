// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include "Stream.h"

#define IMPLEMENTATION_STRING "KOKKOS"

#ifdef KOKKOS_TARGET_CPU
  #define DEVICE Kokkos::OpenMP
#else
  #define DEVICE Kokkos::Cuda
#endif

template <class T>
class KOKKOSStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // Device side pointers to arrays
    Kokkos::View<double*, DEVICE>* d_a;
    Kokkos::View<double*, DEVICE>* d_b;
    Kokkos::View<double*, DEVICE>* d_c;
    Kokkos::View<double*>::HostMirror* hm_a;
    Kokkos::View<double*>::HostMirror* hm_b;
    Kokkos::View<double*>::HostMirror* hm_c;

  public:

    KOKKOSStream(const unsigned int, const int);
    ~KOKKOSStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(
            const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(
            std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

