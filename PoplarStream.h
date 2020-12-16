
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <poplar/Engine.hpp>
#include <poplar/Program.hpp>

#include "Stream.h"

#define IMPLEMENTATION_STRING "Poplar"


using namespace poplar::program;

template<class T>
class PoplarStream : public Stream<T> {

protected:
    unsigned int arraySize;
    const bool halfPrecision;
    T sum = 0;
    std::unique_ptr <poplar::Engine> engine;
    poplar::Target target;
    std::unique_ptr<T[]> a;
    std::unique_ptr<T[]> b;
    std::unique_ptr<T[]> c;

public:

    PoplarStream(const unsigned int, const int, const bool halfPrecision);

    ~PoplarStream();

    virtual void copy() override;

    virtual void add() override;

    virtual void mul() override;

    virtual void triad() override;

    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;

    virtual void read_arrays(std::vector <T> &a, std::vector <T> &b, std::vector <T> &c) override;
    virtual void copyArrays(const T *src, T *dst);

};

