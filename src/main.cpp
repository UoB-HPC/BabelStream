
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>

#include "common.h"
#include "Stream.h"
#include "CUDAStream.h"


const unsigned int ARRAY_SIZE = 52428800;
const unsigned int ntimes = 10;

#define IMPLEMENTATION_STRING "CUDA"

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c);

int main(int argc, char *argv[])
{
  std::cout
    << "GPU-STREAM" << std::endl
    << "Version:" << VERSION_STRING << std::endl
    << "Implementation: " << IMPLEMENTATION_STRING << std::endl;


  // Create host vectors
  std::vector<double> a(ARRAY_SIZE, 1.0);
  std::vector<double> b(ARRAY_SIZE, 2.0);
  std::vector<double> c(ARRAY_SIZE, 0.0);

  Stream<double> *stream;

  // Use the CUDA implementation
  stream = new CUDAStream<double>(ARRAY_SIZE);

  stream->write_arrays(a, b, c);

  // List of times
  std::vector<double> copy_timings;
  std::vector<double> mul_timings;
  std::vector<double> add_timings;
  std::vector<double> triad_timings;

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Main loop
  for (unsigned int k = 0; k < ntimes; k++)
  {

    // Execute Copy
    t1 = std::chrono::high_resolution_clock::now();
    stream->copy();
    t2 = std::chrono::high_resolution_clock::now();
    copy_timings.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Mul
    t1 = std::chrono::high_resolution_clock::now();
    stream->mul();
    t2 = std::chrono::high_resolution_clock::now();
    mul_timings.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Add
    t1 = std::chrono::high_resolution_clock::now();
    stream->add();
    t2 = std::chrono::high_resolution_clock::now();
    add_timings.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Triad
    t1 = std::chrono::high_resolution_clock::now();
    stream->triad();
    t2 = std::chrono::high_resolution_clock::now();
    triad_timings.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

  }

  // Check solutions
  stream->read_arrays(a, b, c);
  check_solution<double>(ntimes, a, b, c);

  // Crunch timing results

  // Get min/max; ignore first result
  auto copy_minmax = std::minmax_element(copy_timings.begin()+1, copy_timings.end());
  auto mul_minmax = std::minmax_element(mul_timings.begin()+1, mul_timings.end());
  auto add_minmax = std::minmax_element(add_timings.begin()+1, add_timings.end());
  auto triad_minmax = std::minmax_element(triad_timings.begin()+1, triad_timings.end());

  double copy_average = std::accumulate(copy_timings.begin()+1, copy_timings.end(), 0.0) / (double)(ntimes - 1);
  double mul_average = std::accumulate(mul_timings.begin()+1, mul_timings.end(), 0.0) / (double)(ntimes - 1);
  double add_average = std::accumulate(add_timings.begin()+1, add_timings.end(), 0.0) / (double)(ntimes - 1);
  double triad_average = std::accumulate(triad_timings.begin()+1, triad_timings.end(), 0.0) / (double)(ntimes - 1);


  // Display results
  std::cout
    << std::left << std::setw(12) << "Function"
    << std::left << std::setw(12) << "MBytes/sec"
    << std::left << std::setw(12) << "Min (sec)"
    << std::left << std::setw(12) << "Max"
    << std::left << std::setw(12) << "Average" << std::endl;

  std::cout << std::fixed;

  std::cout
    << std::left << std::setw(12) << "Copy"
    << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * (2 * sizeof(double) * ARRAY_SIZE)/(*copy_minmax.first)
    << std::left << std::setw(12) << std::setprecision(5) << *copy_minmax.first
    << std::left << std::setw(12) << std::setprecision(5) << *copy_minmax.second
    << std::left << std::setw(12) << std::setprecision(5) << copy_average
    << std::endl;


  std::cout
    << std::left << std::setw(12) << "Mul"
    << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * (2 * sizeof(double) * ARRAY_SIZE)/(*mul_minmax.first)
    << std::left << std::setw(12) << std::setprecision(5) << *mul_minmax.first
    << std::left << std::setw(12) << std::setprecision(5) << *mul_minmax.second
    << std::left << std::setw(12) << std::setprecision(5) << mul_average
    << std::endl;


  std::cout
    << std::left << std::setw(12) << "Add"
    << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * (3 * sizeof(double) * ARRAY_SIZE)/(*add_minmax.first)
    << std::left << std::setw(12) << std::setprecision(5) << *add_minmax.first
    << std::left << std::setw(12) << std::setprecision(5) << *add_minmax.second
    << std::left << std::setw(12) << std::setprecision(5) << add_average
    << std::endl;

  std::cout
    << std::left << std::setw(12) << "Triad"
    << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * (3 * sizeof(double) * ARRAY_SIZE)/(*triad_minmax.first)
    << std::left << std::setw(12) << std::setprecision(5) << *triad_minmax.first
    << std::left << std::setw(12) << std::setprecision(5) << *triad_minmax.second
    << std::left << std::setw(12) << std::setprecision(5) << triad_average
    << std::endl;

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

