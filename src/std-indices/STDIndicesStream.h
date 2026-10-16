// Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
// Updated 2021 by University of Bristol
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <iterator>
#include <type_traits>
#include "Stream.h"

#define IMPLEMENTATION_STRING "STD (index-oriented)"


// A lightweight counting iterator which will be used by the STL algorithms
// NB: C++ <= 17 doesn't have this built-in, and it's only added later in ranges-v3 (C++2a) which this
// implementation doesn't target
template <typename N>
class ranged {
public:
  class iterator {
    friend class ranged;
    public:
    using difference_type = N;
    using value_type = N;
    using pointer = const N*;
    using reference = N;
    using iterator_category = std::random_access_iterator_tag;

    // XXX This is not part of the iterator spec, it gets picked up by oneDPL if enabled.
    // Without this, the DPL SYCL backend collects the iterator data on the host and copies to the device.
    // This type is unused for any nother STL impl.
    using is_passed_directly = std::true_type;

    reference operator *() const { return i_; }
    iterator &operator ++() { ++i_; return *this; }
    iterator operator ++(int) { iterator copy(*this); ++i_; return copy; }

    iterator &operator --() { --i_; return *this; }
    iterator operator --(int) { iterator copy(*this); --i_; return copy; }

    iterator &operator +=(N by) { i_+=by; return *this; }

    value_type operator[](const difference_type &i) const { return i_ + i; }

    difference_type operator-(const iterator &it) const { return i_ - it.i_; }
    iterator operator+(const value_type v) const { return iterator(i_ + v); }

    bool operator ==(const iterator &other) const { return i_ == other.i_; }
    bool operator !=(const iterator &other) const { return i_ != other.i_; }
    bool operator < (const iterator &other) const { return i_ < other.i_; }

    protected:
      explicit iterator(N start) : i_ (start) {}

    private:
        N i_;
  };

  [[nodiscard]] iterator begin() const { return begin_; }
  [[nodiscard]] iterator end() const { return end_; }
  ranged(N begin, N end) : begin_(begin), end_(end) {}
private:
  iterator begin_;
  iterator end_;
};

template <class T>
class STDIndicesStream : public Stream<T>
{
  protected:
    // Size of arrays
    intptr_t array_size;

    // induction range
    ranged<intptr_t> range;

    // Device side pointers
    T *a, *b, *c;

  public:
    STDIndicesStream(BenchId bs, const intptr_t array_size, const int device_id,
		  T initA, T initB, T initC) noexcept;
    ~STDIndicesStream();

    void copy() override;
    void add() override;
    void mul() override;
    void triad() override;
    void nstream() override;
    T dot() override;

    void get_arrays(T const*& a, T const*& b, T const*& c) override;
    void init_arrays(T initA, T initB, T initC) override;
};
