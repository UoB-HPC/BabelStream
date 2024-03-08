
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#define VERSION_STRING "5.0"

#include "Stream.h"
#include "StreamModels.h"

// Default size of 2^25
int ARRAY_SIZE = 33554432;
unsigned int num_times = 100;
unsigned int deviceIndex = 0;
bool use_float = false;
bool output_as_csv = false;
bool mibibytes = false;
std::string csv_separator = ",";

// Benchmarks:
constexpr size_t num_benchmarks = 6;
array<char const*, num_benchmarks> labels = {"Copy", "Add", "Mul", "Triad", "Dot", "Nstream"};
// Weights data moved by benchmark & therefore achieved BW:
// bytes = weight * sizeof(T) * ARRAY_SIZE -> bw = bytes / dur
array<size_t, num_benchmarks> weight = {/*Copy:*/ 2, /*Add:*/ 2, /*Mul:*/ 3, /*Triad:*/ 3, /*Dot:*/ 2, /*Nstream:*/ 4};

// Options for running the benchmark:
// - All 5 kernels (Copy, Add, Mul, Triad, Dot).
// - Triad only.
// - Nstream only.
enum class Benchmark {All, Triad, Nstream};

// Selected run options.
Benchmark selection = Benchmark::All;

// Clock and duration types:
using clk_t = chrono::high_resolution_clock;
using dur_t = chrono::duration<double>;

// Returns duration of executing function f:
template <typename F>
double time(F&& f) {
  auto start = clk_t::now();
  f();
  return dur_t(clk_t::now() - start).count();
}

template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum);

template <typename T>
void run();

// Units for output:
enum class Unit { Mega, Giga };

void parseArguments(int argc, char *argv[]);

int main(int argc, char *argv[])
{

  parseArguments(argc, argv);

  if (!output_as_csv)
  {
    std::cout
      << "BabelStream" << std::endl
      << "Version: " << VERSION_STRING << std::endl
      << "Implementation: " << IMPLEMENTATION_STRING << std::endl;
  }

  if (use_float)
    run<float>();
  else
    run<double>();

  return 0;
}


// Run the 5 main kernels
template <typename T>

std::vector<std::vector<double>> run_all(std::unique_ptr<Stream<T>>& stream, T& sum)
{

  // List of times
  std::vector<std::vector<double>> timings(5);

  // Main loop
  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Copy
    timings[0].push_back(time([&] { stream->copy(); }));

    // Execute Mul
    timings[1].push_back(time([&] { stream->mul(); }));

    // Execute Add
    timings[2].push_back(time([&] { stream->add(); }));

    // Execute Triad
    timings[3].push_back(time([&] { stream->triad(); }));

    // Execute Dot
    timings[4].push_back(time([&] { sum = stream->dot(); }));
  }

  // Compiler should use a move
  return timings;
}

// Run the Triad kernel
template <typename T>
std::vector<std::vector<double>> run_triad(std::unique_ptr<Stream<T>>& stream)
{

  std::vector<std::vector<double>> timings(1);

  // Run triad in loop (compute a single time only!):
  timings[0].push_back(time([&] {
    for (int k = 0; k < num_times; k++) stream->triad();
  }));

  return timings;
}

// Run the Nstream kernel
template <typename T>
std::vector<std::vector<double>> run_nstream(std::unique_ptr<Stream<T>>& stream)
{
  std::vector<std::vector<double>> timings(1);

  // Run nstream in loop
  for (int k = 0; k < num_times; k++) {
    timings[0].push_back(time([&] { stream->nstream(); }));
  }

  return timings;
}

// Generic run routine
// Runs the kernel(s) and prints output.
template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  if (!output_as_csv)
  {
    if (selection == Benchmark::All)
      std::cout << "Running kernels " << num_times << " times" << std::endl;
    else if (selection == Benchmark::Triad)
    {
      std::cout << "Running triad " << num_times << " times" << std::endl;
      std::cout << "Number of elements: " << ARRAY_SIZE << std::endl;
    }


    if (sizeof(T) == sizeof(float))
      std::cout << "Precision: float" << std::endl;
    else
      std::cout << "Precision: double" << std::endl;


    if (mibibytes)
    {
      // MiB = 2^20
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*std::pow(2.0, -20.0) << " MiB"
                << " (=" << ARRAY_SIZE*sizeof(T)*std::pow(2.0, -30.0) << " GiB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*std::pow(2.0, -20.0) << " MiB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*std::pow(2.0, -30.0) << " GiB)" << std::endl;
    }
    else
    {
      // MB = 10^6
      std::cout << std::setprecision(1) << std::fixed
                << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
      std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
                << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
    }
    std::cout.precision(ss);

  }

  auto stream = construct_stream<T>(ARRAY_SIZE, deviceIndex);

  auto initElapsedS = time([&] { stream->init_arrays(startA, startB, startC); });

  // Result of the Dot kernel, if used.
  T sum{};

  std::vector<std::vector<double>> timings;

  switch (selection)
  {
    case Benchmark::All:
      timings = run_all<T>(stream, sum);
      break;
    case Benchmark::Triad:
      timings = run_triad<T>(stream);
      break;
    case Benchmark::Nstream:
      timings = run_nstream<T>(stream);
      break;
  };

  // Check solutions
  // Create host vectors
  std::vector<T> a(ARRAY_SIZE);
  std::vector<T> b(ARRAY_SIZE);
  std::vector<T> c(ARRAY_SIZE);

  auto readElapsedS = time([&] { stream->read_arrays(a, b, c); });

  check_solution<T>(num_times, a, b, c, sum);

  auto fmt_bw = [&](size_t weight, double dt, Unit unit = Unit::Mega) {
    double bps = (weight * sizeof(T) * ARRAY_SIZE)/dt;
    switch(unit) {
    case Unit::Mega: return (mibibytes ? pow(2.0, -20.0) : 1.0E-6) * bps;
    case Unit::Giga: return (mibibytes ? pow(2.0, -30.0) : 1.0E-9) * bps;
    default: cerr << "Unimplemented!" << endl; abort();
    }
  };

  auto initBWps = fmt_bw(3, initElapsedS);
  auto readBWps = fmt_bw(3, readElapsedS);

  if (output_as_csv)
  {
    std::cout
      << "phase" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec") << csv_separator
      << "runtime" << std::endl;
    std::cout
      << "Init" << csv_separator
      << ARRAY_SIZE << csv_separator
      << sizeof(T) << csv_separator
      << initBWps << csv_separator
      << initElapsedS << std::endl;
    std::cout
      << "Read" << csv_separator
      << ARRAY_SIZE << csv_separator
      << sizeof(T) << csv_separator
      << readBWps << csv_separator
      << readElapsedS << std::endl;
  }
  else
  {
    std::cout << "Init: "
      << std::setw(7)
      << initElapsedS
      << " s (="
      << initBWps
      << (mibibytes ? " MiBytes/sec" : " MBytes/sec")
      << ")" << std::endl;
    std::cout << "Read: "
      << std::setw(7)
      << readElapsedS
      << " s (="
      << readBWps
      << (mibibytes ? " MiBytes/sec" : " MBytes/sec")
      << ")" << std::endl;
  }

  // Display timing results
  if (output_as_csv)
  {
    std::cout
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec") << csv_separator
      << "min_runtime" << csv_separator
      << "max_runtime" << csv_separator
      << "avg_runtime" << std::endl;
  }
  else
  {
    std::cout
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << ((mibibytes) ? "MiBytes/sec" : "MBytes/sec")
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << std::fixed;
  }


  if (selection == Benchmark::All || selection == Benchmark::Nstream)
  {
    for (int i = 0; i < timings.size(); ++i)
    {
      // Get min/max; ignore the first result
      auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

      // Calculate average; ignore the first result
      double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

      // Display results
      if (output_as_csv)
      {
        std::cout
          << labels[i] << csv_separator
          << num_times << csv_separator
          << ARRAY_SIZE << csv_separator
          << sizeof(T) << csv_separator
          << fmt_bw(weight[i], *minmax.first) << csv_separator
          << *minmax.first << csv_separator
          << *minmax.second << csv_separator
          << average
          << std::endl;
      }
      else
      {

        std::cout
          << std::left << std::setw(12) << labels[i]
          << std::left << std::setw(12) << std::setprecision(3) << fmt_bw(weight[i], *minmax.first)
          << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
          << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
          << std::left << std::setw(12) << std::setprecision(5) << average
          << std::endl;
          << left << setw(12) << setprecision(3) 
      }
    }
  } else if (selection == Benchmark::Triad)
  {
    // Display timing results
    double total_bytes = 3 * sizeof(T) * ARRAY_SIZE * num_times;
    double bandwidth = fmt_bw(3 * num_times, timings[0][0], Unit::Giga);

    if (output_as_csv)
    {
      std::cout
        << "function" << csv_separator
        << "num_times" << csv_separator
        << "n_elements" << csv_separator
        << "sizeof" << csv_separator
        << ((mibibytes) ? "gibytes_per_sec" : "gbytes_per_sec") << csv_separator
        << "runtime"
        << std::endl;
      std::cout
        << "Triad" << csv_separator
        << num_times << csv_separator
        << ARRAY_SIZE << csv_separator
        << sizeof(T) << csv_separator
        << bandwidth << csv_separator
        << timings[0][0]
        << std::endl;
    }
    else
    {
      std::cout
        << "--------------------------------"
        << std::endl << std::fixed
        << "Runtime (seconds): " << std::left << std::setprecision(5)
        << timings[0][0] << std::endl
        << "Bandwidth (" << ((mibibytes) ? "GiB/s" : "GB/s") << "):  "
        << std::left << std::setprecision(3)
        << bandwidth << std::endl;
    }
  }
}


template <typename T>
void check_solution(const unsigned int ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum)
{
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;
  T goldSum{};

  const T scalar = startScalar;

  for (unsigned int i = 0; i < ntimes; i++)
  {
    // Do STREAM!
    if (selection == Benchmark::All)
    {
      goldC = goldA;
      goldB = scalar * goldC;
      goldC = goldA + goldB;
      goldA = goldB + scalar * goldC;
    } else if (selection == Benchmark::Triad)
    {
      goldA = goldB + scalar * goldC;
    } else if (selection == Benchmark::Nstream)
    {
      goldA += goldB + scalar * goldC;
    }
  }

  // Do the reduction
  goldSum = goldA * goldB * ARRAY_SIZE;

  // Calculate the average error
  long double errA = std::accumulate(a.begin(), a.end(), T{}, [&](double sum, const T val){ return sum + std::fabs(val - goldA); });
  errA /= a.size();
  long double errB = std::accumulate(b.begin(), b.end(), T{}, [&](double sum, const T val){ return sum + std::fabs(val - goldB); });
  errB /= b.size();
  long double errC = std::accumulate(c.begin(), c.end(), T{}, [&](double sum, const T val){ return sum + std::fabs(val - goldC); });
  errC /= c.size();
  long double errSum = std::fabs((sum - goldSum)/goldSum);

  long double epsi = std::numeric_limits<T>::epsilon() * 100.0;

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
  // Check sum to 8 decimal places
  if (selection == Benchmark::All && errSum > 1.0E-8)
    std::cerr
      << "Validation failed on sum. Error " << errSum
      << std::endl << std::setprecision(15)
      << "Sum was " << sum << " but should be " << goldSum
      << std::endl;

}

int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int parseInt(const char *str, int *output)
{
  char *next;
  *output = strtol(str, &next, 10);
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
      if (++i >= argc || !parseInt(argv[i], &ARRAY_SIZE) || ARRAY_SIZE <= 0)
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
    else if (!std::string("--triad-only").compare(argv[i]))
    {
      selection = Benchmark::Triad;
    }
    else if (!std::string("--nstream-only").compare(argv[i]))
    {
      selection = Benchmark::Nstream;
    }
    else if (!std::string("--csv").compare(argv[i]))
    {
      output_as_csv = true;
    }
    else if (!std::string("--mibibytes").compare(argv[i]))
    {
      mibibytes = true;
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
      std::cout << "      --triad-only         Only run triad" << std::endl;
      std::cout << "      --nstream-only       Only run nstream" << std::endl;
      std::cout << "      --csv                Output as csv table" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
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
