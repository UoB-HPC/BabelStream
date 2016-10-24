
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

#include "common.h"
#include "Stream.h"

#if defined(CUDA)
#include "CUDAStream.h"
#elif defined(HIP)
#include "HIPStream.h"
#elif defined(OCL)
#include "OCLStream.h"
#elif defined(USE_RAJA)
#include "RAJAStream.hpp"
#elif defined(KOKKOS)
#include "KOKKOSStream.hpp"
#elif defined(ACC)
#include "ACCStream.h"
#elif defined(SYCL)
#include "SYCLStream.h"
#elif defined(OMP3)
#include "OMP3Stream.h"
#elif defined(OMP45)
#include "OMP45Stream.h"
#endif

// Default size of 2^25
unsigned int ARRAY_SIZE = 33554432;
unsigned int num_times = 100;
unsigned int deviceIndex = 0;
bool use_float = false;

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c);

template <typename T>
void run();

void parseArguments(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  std::cout
    << "GPU-STREAM" << std::endl
    << "Version: " << VERSION_STRING << std::endl
    << "Implementation: " << IMPLEMENTATION_STRING << std::endl;

  parseArguments(argc, argv);

  // TODO: Fix SYCL to allow multiple template specializations
#ifndef SYCL
#ifndef KOKKOS
  if (use_float)
    run<float>();
  else
#endif
#endif
    run<double>();

}

template <typename T>
void run()
{
  std::cout << "Running kernels " << num_times << " times" << std::endl;

  if (sizeof(T) == sizeof(float))
    std::cout << "Precision: float" << std::endl;
  else
    std::cout << "Precision: double" << std::endl;

  // Create host vectors
  std::vector<T> a(ARRAY_SIZE, startA);
  std::vector<T> b(ARRAY_SIZE, startB);
  std::vector<T> c(ARRAY_SIZE, startC);
  std::streamsize ss = std::cout.precision();
  std::cout << std::setprecision(1) << std::fixed
    << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout.precision(ss);

  Stream<T> *stream;

#if defined(CUDA)
  // Use the CUDA implementation
  stream = new CUDAStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(HIP)
  // Use the HIP implementation
  stream = new HIPStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(OCL)
  // Use the OpenCL implementation
  stream = new OCLStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(USE_RAJA)
  // Use the RAJA implementation
  stream = new RAJAStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(KOKKOS)
  // Use the Kokkos implementation
  stream = new KOKKOSStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(ACC)
  // Use the OpenACC implementation
  stream = new ACCStream<T>(ARRAY_SIZE, a.data(), b.data(), c.data(), deviceIndex);

#elif defined(SYCL)
  // Use the SYCL implementation
  stream = new SYCLStream<T>(ARRAY_SIZE, deviceIndex);

#elif defined(OMP3)
  // Use the "reference" OpenMP 3 implementation
  stream = new OMP3Stream<T>(ARRAY_SIZE, a.data(), b.data(), c.data());

#elif defined(OMP45)
  // Use the "reference" OpenMP 3 implementation
  stream = new OMP45Stream<T>(ARRAY_SIZE, a.data(), b.data(), c.data(), deviceIndex);

#endif

  stream->write_arrays(a, b, c);

  // List of times
  std::vector<std::vector<double>> timings(4);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Copy
    t1 = std::chrono::high_resolution_clock::now();
    stream->copy();
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Mul
    t1 = std::chrono::high_resolution_clock::now();
    stream->mul();
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Add
    t1 = std::chrono::high_resolution_clock::now();
    stream->add();
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Triad
    t1 = std::chrono::high_resolution_clock::now();
    stream->triad();
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

  }

  // Check solutions
  stream->read_arrays(a, b, c);
  check_solution<T>(num_times, a, b, c);

  // Display timing results
  std::cout
    << std::left << std::setw(12) << "Function"
    << std::left << std::setw(12) << "MBytes/sec"
    << std::left << std::setw(12) << "Min (sec)"
    << std::left << std::setw(12) << "Max"
    << std::left << std::setw(12) << "Average" << std::endl;

  std::cout << std::fixed;

  std::string labels[4] = {"Copy", "Mul", "Add", "Triad"};
  size_t sizes[4] = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE
  };

  for (int i = 0; i < 4; i++)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

    // Display results
    std::cout
      << std::left << std::setw(12) << labels[i]
      << std::left << std::setw(12) << std::setprecision(3) << 1.0E-6 * sizes[i] / (*minmax.first)
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
      << std::left << std::setw(12) << std::setprecision(5) << average
      << std::endl;

  }

  delete stream;

}

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;

  const T scalar = startScalar;

  for (unsigned int i = 0; i < ntimes; i++)
  {
    // Do STREAM!
    goldC = goldA;
    goldB = scalar * goldC;
    goldC = goldA + goldB;
    goldA = goldB + scalar * goldC;
  }

  // Calculate the average error
  double errA = std::accumulate(a.begin(), a.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldA); });
  errA /= a.size();
  double errB = std::accumulate(b.begin(), b.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldB); });
  errB /= b.size();
  double errC = std::accumulate(c.begin(), c.end(), 0.0, [&](double sum, const T val){ return sum + fabs(val - goldC); });
  errC /= c.size();

  double epsi = std::numeric_limits<T>::epsilon() * 100.0;

  if (errA > epsi)
    std::cerr
      << "Validation failed on a[]. Average error " << errA
      << std::endl;
  if (errB > epsi)
    std::cerr
      << "Validation failed on b[]. Average error " << errB
      << std::endl;
  if (errC > epsi)
    std::cerr
      << "Validation failed on c[]. Average error " << errC
      << std::endl;

}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--list").compare(argv[i]))
    {
      listDevices();
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--device").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &deviceIndex))
      {
        std::cerr << "Invalid device index." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--arraysize").compare(argv[i]) ||
             !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &ARRAY_SIZE))
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
             !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--float").compare(argv[i]))
    {
      use_float = true;
    }
    else if (!std::string("--help").compare(argv[i]) ||
             !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "      --list               List available devices" << std::endl;
      std::cout << "      --device     INDEX   Select device at INDEX" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << "      --float              Use floats (rather than doubles)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}
