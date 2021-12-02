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

#define IMPLEMENTATION_STRING "Kokkos"

template <class T>
class KokkosStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Device side pointers to arrays
     typename Kokkos::View<T*>* d_a;
     typename Kokkos::View<T*>* d_b;
     typename Kokkos::View<T*>* d_c;
     typename Kokkos::View<T*>::HostMirror* hm_a;
     typename Kokkos::View<T*>::HostMirror* hm_b;
     typename Kokkos::View<T*>::HostMirror* hm_c;

  public:

    KokkosStream(const int, const int);
    ~KokkosStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(
            std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

