// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include "Stream.h"

#define IMPLEMENTATION_STRING "Kokkos"

template <class T>
class KokkosStream : public Stream<T>
{
  protected:
    // Size of arrays
    intptr_t array_size;

    // Device side pointers to arrays
    typename Kokkos::View<T*>* d_a;
    typename Kokkos::View<T*>* d_b;
    typename Kokkos::View<T*>* d_c;
    typename Kokkos::View<T*>::HostMirror* hm_a;
    typename Kokkos::View<T*>::HostMirror* hm_b;
    typename Kokkos::View<T*>::HostMirror* hm_c;

  public:

    KokkosStream(BenchId bs, const intptr_t array_size, const int device_id,
		 T initA, T initB, T initC);
    ~KokkosStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC);
};

