
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"
#include "hc.hpp"

#define IMPLEMENTATION_STRING "HC"

template <class T>
class HCStream : public Stream<T>
{
protected:
  // Size of arrays
  int array_size;
  // Device side pointers to arrays
  hc::array<T,1> d_a;
  hc::array<T,1> d_b;
  hc::array<T,1> d_c;


public:

  HCStream(const int, const int);
  ~HCStream();

  virtual void copy() override;
  virtual void add() override;
  virtual void mul() override;
  virtual void triad() override;
  virtual T dot() override;
  T dot_impl();

  virtual void init_arrays(T initA, T initB, T initC) override;
  virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};
