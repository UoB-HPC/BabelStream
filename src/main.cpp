
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>

#include "common.h"
#include "Stream.h"
#include "CUDAStream.h"


const unsigned int ARRAY_SIZE = 52428800;

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

  stream->copy();
  stream->mul();
  stream->add();
  stream->triad();

  stream->read_arrays(a, b, c);
  std::cout << a[105] << std::endl;

  check_solution<double>(1, a, b, c);

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

