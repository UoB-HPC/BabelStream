#include "hip_runtime.h"
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

//#include <cuda.h>
#include "common.h"

std::string getDeviceName(int device);
int getDriver(void);

// Code to check CUDA errors
void check_cuda_error(void)
{
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::cerr
            << "Error: "
            << hipGetErrorString(err)
            << std::endl;
            exit(err);
    }
}



// looper function place more work inside each work item.
// Goal is reduce the dispatch overhead for each group, and also give more controlover the order of memory operations
template <typename T>
__global__ void
copy_looper(hipLaunchParm lp,  const T * a, T * c, int ARRAY_SIZE)
{
    int offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int stride = hipBlockDim_x * hipGridDim_x;

    for (int i=offset; i<ARRAY_SIZE; i+=stride) {
        c[i] = a[i];
    }
}

template <typename T>
__global__ void
mul_looper(hipLaunchParm lp,  T * b, const T * c, int ARRAY_SIZE)
{
    int offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int stride = hipBlockDim_x * hipGridDim_x;
    const T scalar = 3.0;

    for (int i=offset; i<ARRAY_SIZE; i+=stride) {
        b[i] = scalar * c[i];
    }
}

template <typename T>
__global__ void
add_looper(hipLaunchParm lp,  const T * a, const T * b, T * c, int ARRAY_SIZE)
{
    int offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int stride = hipBlockDim_x * hipGridDim_x;

    for (int i=offset; i<ARRAY_SIZE; i+=stride) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
__global__ void
triad_looper(hipLaunchParm lp,  T * a, const T * b, const T * c, int ARRAY_SIZE)
{
    int offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int stride = hipBlockDim_x * hipGridDim_x;
    const T scalar = 3.0;

    for (int i=offset; i<ARRAY_SIZE; i+=stride) {
        a[i] = b[i] + scalar * c[i];
    }
}




template <typename T>
__global__ void
copy(hipLaunchParm lp,  const T * a, T * c)
{
    const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    c[i] = a[i];
}


template <typename T>
__global__ void
mul(hipLaunchParm lp,  T * b, const T * c)
{
    const T scalar = 3.0;
    const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    b[i] = scalar * c[i];
}

template <typename T>
__global__ void
add(hipLaunchParm lp,  const T * a, const T * b, T * c)
{
    const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    c[i] = a[i] + b[i];
}

template <typename T>
__global__ void
triad(hipLaunchParm lp,  T * a, const T * b, const T * c)
{
    const T scalar = 3.0;
    const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    a[i] = b[i] + scalar * c[i];
}

int main(int argc, char *argv[])
{

    // Print out run information
    std::cout
        << "GPU-STREAM" << std::endl
        << "Version: " << VERSION_STRING << std::endl
        << "Implementation: HIP" << std::endl;

    parseArguments(argc, argv);

    if (NTIMES < 2)
        throw std::runtime_error("Chosen number of times is invalid, must be >= 2");

    // Config grid size and group size for kernel launching
    int gridSize;
    if (groups) {
        gridSize = groups * groupSize;
    } else  {
        gridSize = ARRAY_SIZE;
    }

    float operationsPerWorkitem = (float)ARRAY_SIZE / (float)gridSize;
    std::cout << "GridSize: " << gridSize << " work-items" << std::endl;
    std::cout << "GroupSize: " << groupSize << " work-items" << std::endl;
    std::cout << "Operations/Work-item: " << operationsPerWorkitem << std::endl;
    if (groups) std::cout << "Using looper kernels:" << std::endl;

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
        if (ARRAY_SIZE == 0)
            throw std::runtime_error("Array size must be >= 1024");
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
		<< " " << ARRAY_PAD_BYTES << " bytes padding"
        << std::endl;
    std::cout << "Total size: " << 3.0*(ARRAY_SIZE*DATATYPE_SIZE + ARRAY_PAD_BYTES) /1024.0/1024.0 << " MB"
        << " (=" << 3.0*(ARRAY_SIZE*DATATYPE_SIZE + ARRAY_PAD_BYTES) /1024.0/1024.0/1024.0 << " GB)"
        << std::endl;

    // Reset precision
    std::cout.precision(ss);

    // Check device index is in range
    int count;
    hipGetDeviceCount(&count);
    check_cuda_error();
    if (deviceIndex >= count)
        throw std::runtime_error("Chosen device index is invalid");
    hipSetDevice(deviceIndex);
    check_cuda_error();


    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, deviceIndex);

    // Print out device name
    std::cout << "Using HIP device " << getDeviceName(deviceIndex) <<  " (compute_units=" << props.multiProcessorCount << ")" << std::endl;

    // Print out device HIP driver version
    std::cout << "Driver: " << getDriver() << std::endl;




    // Check buffers fit on the device
    if (props.totalGlobalMem < 3*DATATYPE_SIZE*ARRAY_SIZE)
        throw std::runtime_error("Device does not have enough memory for all 3 buffers");

    //int cus = props.multiProcessorCount;

    // Create host vectors
    void * h_a = malloc(ARRAY_SIZE*DATATYPE_SIZE );
    void * h_b = malloc(ARRAY_SIZE*DATATYPE_SIZE );
    void * h_c = malloc(ARRAY_SIZE*DATATYPE_SIZE );

    // Initialise arrays
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
    char * d_a, * d_b, *d_c;
    hipMalloc(&d_a, ARRAY_SIZE*DATATYPE_SIZE + ARRAY_PAD_BYTES);
    check_cuda_error();
    hipMalloc(&d_b, ARRAY_SIZE*DATATYPE_SIZE + ARRAY_PAD_BYTES);
    d_b += ARRAY_PAD_BYTES;
    check_cuda_error();
    hipMalloc(&d_c, ARRAY_SIZE*DATATYPE_SIZE + ARRAY_PAD_BYTES);
    d_c += ARRAY_PAD_BYTES;
    check_cuda_error();

    // Copy host memory to device
    hipMemcpy(d_a, h_a, ARRAY_SIZE*DATATYPE_SIZE, hipMemcpyHostToDevice);
    check_cuda_error();
    hipMemcpy(d_b, h_b, ARRAY_SIZE*DATATYPE_SIZE, hipMemcpyHostToDevice);
    check_cuda_error();
    hipMemcpy(d_c, h_c, ARRAY_SIZE*DATATYPE_SIZE, hipMemcpyHostToDevice);
    check_cuda_error();


    std::cout << "d_a=" << (void*)d_a << std::endl;
	std::cout << "d_b=" << (void*)d_b << std::endl;
	std::cout << "d_c=" << (void*)d_c << std::endl;

    // Make sure the copies are finished
    hipDeviceSynchronize();
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
        if (groups) {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(copy_looper<float>), dim3(gridSize), dim3(groupSize), 0, 0, (float*)d_a, (float*)d_c, ARRAY_SIZE);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(copy_looper<double>), dim3(gridSize), dim3(groupSize), 0, 0, (double*)d_a, (double*)d_c, ARRAY_SIZE);
        } else {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(copy), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (float*)d_a, (float*)d_c);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(copy), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (double*)d_a, (double*)d_c);
        }
        check_cuda_error();
        hipDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        if (groups) {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(mul_looper), dim3(gridSize), dim3(groupSize), 0, 0, (float*)d_b, (float*)d_c, ARRAY_SIZE);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(mul_looper), dim3(gridSize), dim3(groupSize), 0, 0, (double*)d_b, (double*)d_c, ARRAY_SIZE);
        } else {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(mul), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (float*)d_b, (float*)d_c);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(mul), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (double*)d_b, (double*)d_c);
        }
        check_cuda_error();
        hipDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        if (groups) {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(add_looper), dim3(gridSize), dim3(groupSize), 0, 0, (float*)d_a, (float*)d_b, (float*)d_c, ARRAY_SIZE);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(add_looper), dim3(gridSize), dim3(groupSize), 0, 0, (double*)d_a, (double*)d_b, (double*)d_c, ARRAY_SIZE);
        } else {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(add), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (float*)d_a, (float*)d_b, (float*)d_c);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(add), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (double*)d_a, (double*)d_b, (double*)d_c);
        }
        check_cuda_error();
        hipDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


        t1 = std::chrono::high_resolution_clock::now();
        if (groups) {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(triad_looper), dim3(gridSize), dim3(groupSize), 0, 0, (float*)d_a, (float*)d_b, (float*)d_c, ARRAY_SIZE);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(triad_looper), dim3(gridSize), dim3(groupSize), 0, 0, (double*)d_a, (double*)d_b, (double*)d_c, ARRAY_SIZE);
        } else {
            if (useFloat)
                hipLaunchKernel(HIP_KERNEL_NAME(triad), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (float*)d_a, (float*)d_b, (float*)d_c);
            else
                hipLaunchKernel(HIP_KERNEL_NAME(triad), dim3(ARRAY_SIZE/groupSize), dim3(groupSize), 0, 0, (double*)d_a, (double*)d_b, (double*)d_c);
        }

        check_cuda_error();
        hipDeviceSynchronize();
        check_cuda_error();
        t2 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

        timings.push_back(times);

    }

    // Check solutions
    hipMemcpy(h_a, d_a, ARRAY_SIZE*DATATYPE_SIZE, hipMemcpyDeviceToHost);
    check_cuda_error();
    hipMemcpy(h_b, d_b, ARRAY_SIZE*DATATYPE_SIZE, hipMemcpyDeviceToHost);
    check_cuda_error();
    hipMemcpy(h_c, d_c, ARRAY_SIZE*DATATYPE_SIZE, hipMemcpyDeviceToHost);
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

    for (int j = 0; j < 4; j++) {
        avg[j] /= (double)(NTIMES-1);
    }

    double geomean = 1.0;
    for (int j = 0; j < 4; j++) {
        geomean *= (sizes[j]/min[j]);
    }
    geomean = pow(geomean, 0.25);

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
    std::cout
        << std::left << std::setw(12) << "GEOMEAN"
        << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * geomean
        << std::endl;

    // Free host vectors
    free(h_a);
    free(h_b);
    free(h_c);

    // Free cuda buffers
    hipFree(d_a);
    check_cuda_error();
    hipFree(d_b);
    check_cuda_error();
    hipFree(d_c);
    check_cuda_error();

}

std::string getDeviceName(int device)
{
    struct hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    check_cuda_error();
    return std::string(prop.name);
}

int getDriver(void)
{
    int driver;
    hipDriverGetVersion(&driver);
    check_cuda_error();
    return driver;
}

void listDevices(void)
{
    // Get number of devices
    int count;
    hipGetDeviceCount(&count);
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

