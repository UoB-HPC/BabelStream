#pragma once

#include <iostream>

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/host_memory_resource.hpp>

#include "Stream.h"

#ifdef VECPAR_GPU
#include "cuda.h"
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#include <vecpar/cuda/cuda_parallelization.hpp>
#endif

#include <vecpar/all/main.hpp>

#if defined(VECPAR_GPU) && defined(MANAGED)
    #define IMPLEMENTATION_STRING "vecpar_gpu_single_source_managed_memory"
    #define SINGLE_SOURCE "1"
#elif defined(VECPAR_GPU) && defined(DEFAULT)
    #define IMPLEMENTATION_STRING "vecpar_gpu_host_device_memory"
#else
//##elif defined(MANAGED)
    #define IMPLEMENTATION_STRING "vecpar_cpu_single_source"
    #define SINGLE_SOURCE "1"
//#else
  //  #define IMPLEMENTATION_STRING "vecpar_cpu_host_memory"
#endif


template <class T>
class VecparStream : public Stream<T>
{
protected:
    // Size of arrays
    int array_size;

    // Host side pointers or managed memory
    vecmem::vector<T> *a;
    vecmem::vector<T> *b;
    vecmem::vector<T> *c;

#if defined(VECPAR_GPU) && defined(MANAGED)
    vecmem::cuda::managed_memory_resource memoryResource;
#elif defined(VECPAR_GPU) && defined(DEFAULT)
    vecmem::host_memory_resource memoryResource;
    vecmem::cuda::device_memory_resource dev_mem;
    vecmem::cuda::copy copy_tool;

    vecmem::data::vector_buffer<T> d_a;
    vecmem::data::vector_buffer<T> d_b;
    vecmem::data::vector_buffer<T> d_c;
#else
    vecmem::host_memory_resource memoryResource;
#endif

public:
    VecparStream(const int, int);
    ~VecparStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};

/// if SHARED MEMORY is set, then vecpar single source code can be used;
/// define one algorithm per function
#ifdef MANAGED
    template <class T>
    struct vecpar_triad :
        public vecpar::algorithm::parallelizable_mmap<
            vecpar::collection::Three,
            vecmem::vector<T>, // a
            vecmem::vector<T>, // b
            vecmem::vector<T>, // c
            const T // scalar
            > {
        TARGET T& map(T& a_i, const T& b_i, const T& c_i, const T scalar) const {
            a_i = b_i + scalar * c_i;
            return a_i;
        }
    };

template <class T>
struct vecpar_add :
        public vecpar::algorithm::parallelizable_mmap<
                vecpar::collection::Three,
                vecmem::vector<T>, // c
                vecmem::vector<T>, // a
                vecmem::vector<T>> // b
                {
    TARGET T& map(T& c_i, const T& a_i, const T& b_i) const {
        c_i = a_i + b_i ;
        return c_i;
    }
};

template <class T>
struct vecpar_mul:
        public vecpar::algorithm::parallelizable_mmap<
                vecpar::collection::Two,
                vecmem::vector<T>, // b
                vecmem::vector<T>, // c
                const T > //  scalar
            {
    TARGET T& map(T& b_i, const T& c_i, const T scalar) const {
        b_i = scalar * c_i ;
        return b_i;
    }
};

template <class T>
struct vecpar_copy:
        public vecpar::algorithm::parallelizable_mmap<
                vecpar::collection::Two,
                vecmem::vector<T>, // c
                vecmem::vector<T>> // a
{
    TARGET T& map(T& c_i, const T& a_i) const {
        c_i = a_i;
        return c_i;
    }
};

template <class T>
struct vecpar_dot:
        public vecpar::algorithm::parallelizable_map_reduce<
                vecpar::collection::Two,
                T, // reduction result
                vecmem::vector<T>, // map result
                vecmem::vector<T>, // a
                vecmem::vector<T>> // b
{
    TARGET T& map(T& result, T& a_i, const T& b_i) const {
        result = a_i * b_i;
        return result;
    }

    TARGET T* reduce(T* result, T& crt) const {
        *result += crt;
        return result;
    }
};

template <class T>
struct vecpar_nstream : public vecpar::algorithm::parallelizable_mmap<
        vecpar::collection::Three,
        vecmem::vector<T>, // a
        vecmem::vector<T>, // b
        vecmem::vector<T>, // c
        const T> // scalar
{
    TARGET T& map(T& a_i, const T& b_i, const T& c_i, const T scalar) const {
        a_i += b_i + scalar * c_i;
        return a_i;
    }

};
#endif

