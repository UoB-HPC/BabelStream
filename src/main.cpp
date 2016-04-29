
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
#elif defined(OCL)
#include "OCLStream.h"
#endif


const unsigned int ARRAY_SIZE = 52428800;
const unsigned int ntimes = 10;
unsigned int deviceIndex = 0;


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

  run<double>();

}

template <typename T>
void run()
{
  // Create host vectors
  std::vector<T> a(ARRAY_SIZE, 1.0);
  std::vector<T> b(ARRAY_SIZE, 2.0);
  std::vector<T> c(ARRAY_SIZE, 0.0);

  Stream<T> *stream;

#if defined(CUDA)
  // Use the CUDA implementation
  stream = new CUDAStream<T>(ARRAY_SIZE);

#elif defined(OCL)
  // Use the OpenCL implementation
  stream = new OCLStream<T>(ARRAY_SIZE);

#endif

  stream->write_arrays(a, b, c);

  // List of times
  std::vector<std::vector<double>> timings(4);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < ntimes; k++)
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
  check_solution<T>(ntimes, a, b, c);

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
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(ntimes - 1);

    // Display results
    std::cout
      << std::left << std::setw(12) << labels[i]
      << std::left << std::setw(12) << std::setprecision(3) << 1.0E-6 * sizes[i] / (*minmax.first)
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
      << std::left << std::setw(12) << std::setprecision(5) << average
      << std::endl;
    

  }

  delete[] stream;

}

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Generate correct solution
  T goldA = 1.0;
  T goldB = 2.0;
  T goldC = 0.0;

  const T scalar = 3.0;

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
  }
}

