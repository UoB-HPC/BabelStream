/*=============================================================================
*------------------------------------------------------------------------------
* Copyright 2015: Tom Deakin, Simon McIntosh-Smith, University of Bristol HPC
* Based on John D. McCalpin’s original STREAM benchmark for CPUs
*------------------------------------------------------------------------------
* License:
*  1. You are free to use this program and/or to redistribute
*     this program.
*  2. You are free to modify this program for your own use,
*     including commercial use, subject to the publication
*     restrictions in item 3.
*  3. You are free to publish results obtained from running this
*     program, or from works that you derive from this program,
*     with the following limitations:
*     3a. In order to be referred to as "GPU-STREAM benchmark results",
*         published results must be in conformance to the GPU-STREAM
*         Run Rules published at
*         http://github.com/UoB-HPC/GPU-STREAM/wiki/Run-Rules
*         and incorporated herein by reference.
*         The copyright holders retain the
*         right to determine conformity with the Run Rules.
*     3b. Results based on modified source code or on runs not in
*         accordance with the GPU-STREAM Run Rules must be clearly
*         labelled whenever they are published.  Examples of
*         proper labelling include:
*         "tuned GPU-STREAM benchmark results" 
*         "based on a variant of the GPU-STREAM benchmark code"
*         Other comparable, clear and reasonable labelling is
*         acceptable.
*     3c. Submission of results to the GPU-STREAM benchmark web site
*         is encouraged, but not required.
*  4. Use of this program or creation of derived works based on this
*     program constitutes acceptance of these licensing restrictions.
*  5. Absolutely no warranty is expressed or implied.
*———————————————————————————————————-----------------------------------------*/


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <cmath>

#include <cuda.h>
#include "common.h"

std::string getDeviceName(int device);

// Code to check CUDA errors
void check_cuda_error(void)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr
            << "Error: "
            << cudaGetErrorString(err)
            << std::endl;
            exit(err);
    }
}

template <typename T>
__global__ void copy(const T * a, T * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i];
}

template <typename T>
__global__ void mul(T * b, const T * c)
{
    const T scalar = 3.0;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    b[i] = scalar * c[i];
}

template <typename T>
__global__ void add(const T * a, const T * b, T * c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

template <typename T>
__global__ void triad(T * a, const T * b, const T * c)
{
    const T scalar = 3.0;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    a[i] = b[i] + scalar * c[i];
}

int main(int argc, char *argv[])
{

    // Print out run information
    std::cout
        << "GPU-STREAM" << std::endl
        << "Version: " << VERSION_STRING << std::endl
        << "Implementation: CUDA" << std::endl;

    parseArguments(argc, argv);

    if (NTIMES < 2) throw badntimes();

    std::cout << "Precision: ";
    if (useFloat) std::cout << "float";
    else std::cout << "double";
    std::cout << std::endl << std::endl;

    std::cout << "Running kernels " << NTIMES << " times" << std::endl;

    if (ARRAY_SIZE % 1024 != 0)
    {
        unsigned int OLD_ARRAY_SIZE = ARRAY_SIZE;
        ARRAY_SIZE -= ARRAY_SIZE % 1024;
        std::cout
            << "Warning: array size must divide 1024" << std::endl
            << "Resizing array from " << OLD_ARRAY_SIZE
            << " to " << ARRAY_SIZE << std::endl;
        if (ARRAY_SIZE == 0) throw badarraysize();
    }

    // Get precision (used to reset later)
    std::streamsize ss = std::cout.precision();

    size_t DATATYPE_SIZE;

    if (useFloat)
    {
        DATATYPE_SIZE = sizeof(float);
    }
    else
    {
        DATATYPE_SIZE = sizeof(double);
    }

    // Display number of bytes in array
    std::cout << std::setprecision(1) << std::fixed
        << "Array size: " << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
        << " (=" << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
        << std::endl;
    std::cout << "Total size: " << 3.0*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
        << " (=" << 3.0*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
        << std::endl;

    // Reset precision
    std::cout.precision(ss);

    // Check device index is in range
    int count;
    cudaGetDeviceCount(&count);
    check_cuda_error();
    if (deviceIndex >= count) throw invaliddevice();
    cudaSetDevice(deviceIndex);
    check_cuda_error();

    // Print out device name
    std::cout << "Using CUDA device " << getDeviceName(deviceIndex) << std::endl;


    // Create host vectors
    void * h_a = malloc(ARRAY_SIZE*DATATYPE_SIZE);
    void * h_b = malloc(ARRAY_SIZE*DATATYPE_SIZE);
    void * h_c = malloc(ARRAY_SIZE*DATATYPE_SIZE);

    // Initilise arrays
    for (unsigned int i = 0; i < ARRAY_SIZE; i++)
    {
        if (useFloat)
        {
            ((float*)h_a)[i] = 1.0f;
            ((float*)h_b)[i] = 2.0f;
            ((float*)h_c)[i] = 0.0f;
        }
        else
        {
            ((double*)h_a)[i] = 1.0;
            ((double*)h_b)[i] = 2.0;
            ((double*)h_c)[i] = 0.0;
        }
    }

    // Create device buffers
    void * d_a, * d_b, *d_c;
    cudaMalloc(&d_a, ARRAY_SIZE*DATATYPE_SIZE);
    check_cuda_error();
    cudaMalloc(&d_b, ARRAY_SIZE*DATATYPE_SIZE);
    check_cuda_error();
    cudaMalloc(&d_c, ARRAY_SIZE*DATATYPE_SIZE);
    check_cuda_error();

    // Copy host memory to device
    cudaMemcpy(d_a, h_a, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyHostToDevice);
    check_cuda_error();
    cudaMemcpy(d_b, h_b, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyHostToDevice);
    check_cuda_error();
    cudaMemcpy(d_c, h_c, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyHostToDevice);
    check_cuda_error();

    // Make sure the copies are finished
    cudaDeviceSynchronize();
    check_cuda_error();

    // List of times
    std::vector< std::vector<double> > timings;

    // Declare timers
    std::chrono::high_resolution_clock::time_point t1, t2;

    // Main loop
    for (unsigned int k = 0; k < NTIMES; k++)
    {
        std::vector<double> times;
        t1 = std::chrono::high_resolution_clock::now();
        if (useFloat)
            copy<<<ARRAY_SIZE/1024, 1024>>>((float*)d_a, (float*)d_c);
        else
            copy<<<ARRAY_SIZE/1024, 1024>>>((double*)d_a, (double*)d_c);
        check_cuda_error();
        cudaDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        if (useFloat)
            mul<<<ARRAY_SIZE/1024, 1024>>>((float*)d_b, (float*)d_c);
        else
            mul<<<ARRAY_SIZE/1024, 1024>>>((double*)d_b, (double*)d_c);
        check_cuda_error();
        cudaDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        if (useFloat)
            add<<<ARRAY_SIZE/1024, 1024>>>((float*)d_a, (float*)d_b, (float*)d_c);
        else
            add<<<ARRAY_SIZE/1024, 1024>>>((double*)d_a, (double*)d_b, (double*)d_c);
        check_cuda_error();
        cudaDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        if (useFloat)
            triad<<<ARRAY_SIZE/1024, 1024>>>((float*)d_a, (float*)d_b, (float*)d_c);
        else
            triad<<<ARRAY_SIZE/1024, 1024>>>((double*)d_a, (double*)d_b, (double*)d_c);
        check_cuda_error();
        cudaDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

        timings.push_back(times);

    }

    // Check solutions
    cudaMemcpy(h_a, d_a, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyDeviceToHost);
    check_cuda_error();
    cudaMemcpy(h_b, d_b, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyDeviceToHost);
    check_cuda_error();
    cudaMemcpy(h_c, d_c, ARRAY_SIZE*DATATYPE_SIZE, cudaMemcpyDeviceToHost);
    check_cuda_error();

    if (useFloat)
    {
        check_solution<float>(h_a, h_b, h_c);
    }
    else
    {
        check_solution<double>(h_a, h_b, h_c);
    }

    // Crunch results
    size_t sizes[4] = {
        2 * DATATYPE_SIZE * ARRAY_SIZE,
        2 * DATATYPE_SIZE * ARRAY_SIZE,
        3 * DATATYPE_SIZE * ARRAY_SIZE,
        3 * DATATYPE_SIZE * ARRAY_SIZE
    };
    double min[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
    double max[4] = {0.0, 0.0, 0.0, 0.0};
    double avg[4] = {0.0, 0.0, 0.0, 0.0};

    // Ignore first result
    for (unsigned int i = 1; i < NTIMES; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            avg[j] += timings[i][j];
            min[j] = std::min(min[j], timings[i][j]);
            max[j] = std::max(max[j], timings[i][j]);
        }
    }

    for (int j = 0; j < 4; j++)
        avg[j] /= (double)(NTIMES-1);

    // Display results
    std::string labels[] = {"Copy", "Mul", "Add", "Triad"};
    std::cout
        << std::left << std::setw(12) << "Function"
        << std::left << std::setw(12) << "MBytes/sec"
        << std::left << std::setw(12) << "Min (sec)"
        << std::left << std::setw(12) << "Max"
        << std::left << std::setw(12) << "Average"
        << std::endl;

    for (int j = 0; j < 4; j++)
    {
        std::cout
            << std::left << std::setw(12) << labels[j]
            << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * sizes[j]/min[j]
            << std::left << std::setw(12) << std::setprecision(5) << min[j]
            << std::left << std::setw(12) << std::setprecision(5) << max[j]
            << std::left << std::setw(12) << std::setprecision(5) << avg[j]
            << std::endl;
    }

    // Free host vectors
    free(h_a);
    free(h_b);
    free(h_c);

}

std::string getDeviceName(int device)
{
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    check_cuda_error();
    return std::string(prop.name);
}

void listDevices(void)
{
    // Get number of devices
    int count;
    cudaGetDeviceCount(&count);
    check_cuda_error();

    // Print device names
    if (count == 0)
    {
        std::cout << "No devices found." << std::endl;
    }
    else
    {
        std::cout << std::endl;
        std::cout << "Devices:" << std::endl;
        for (int i = 0; i < count; i++)
        {
            std::cout << i << ": " << getDeviceName(i) << std::endl;
            check_cuda_error();
        }
        std::cout << std::endl;
    }
}

