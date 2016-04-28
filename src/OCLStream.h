
#include <iostream>
#include <stdexcept>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "cl2.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "OpenCL"

template <class T>
class OCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // Device side pointers to arrays
    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;

    // OpenCL objects
    cl::Context context;
    cl::CommandQueue queue;

    cl::KernelFunctor<cl::Buffer, cl::Buffer> *copy_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer> * mul_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> *add_kernel;
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> *triad_kernel;

  public:

    OCLStream(const unsigned int);
    ~OCLStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

