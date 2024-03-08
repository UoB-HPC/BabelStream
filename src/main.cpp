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
// - Classic 5 kernels (Copy, Add, Mul, Triad, Dot).
// - All kernels (Copy, Add, Mul, Triad, Dot, Nstream).
// - Individual kernels only.
enum class Benchmark : int {Copy = 0, Add = 1, Mul = 2, Triad = 3, Dot = 4, Nstream = 5, Classic, All};

// Selected run options.
Benchmark selection = Benchmark::All;

// Returns true if the benchmark needs to be run:
bool run_benchmark(int id) {
  if (selection == Benchmark::All)                return true;
  if (selection == Benchmark::Classic && id < 5)  return true;
  if (id == 0 && selection == Benchmark::Copy)    return true;
  if (id == 1 && selection == Benchmark::Add)     return true;
  if (id == 2 && selection == Benchmark::Mul)     return true;
  if (id == 3 && selection == Benchmark::Triad)   return true;
  if (id == 4 && selection == Benchmark::Dot)     return true;
  if (id == 5 && selection == Benchmark::Nstream) return true;
  return false;
}

// Prints all available benchmark labels:
template <typename OStream>
void print_labels(OStream& os) {
  for (int i = 0; i < num_benchmarks; ++i) {
    os << labels[i];
    if (i != (num_benchmarks - 1)) os << ",";
  }
}

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

// Run specified kernels
template <typename T>

std::vector<std::vector<double>> run_all(std::unique_ptr<Stream<T>>& stream, T& sum)
{
  // Times for each measured benchmark:
  std::vector<std::vector<double>> timings(num_benchmarks);

  // Time a particular benchmark:
  auto dt = [&](size_t i)
  {
    switch((Benchmark)i) {
    case Benchmark::Copy:    return time([&] { stream->copy(); });
    case Benchmark::Mul:     return time([&] { stream->mul(); });
    case Benchmark::Add:     return time([&] { stream->add(); });
    case Benchmark::Triad:   return time([&] { stream->triad(); });
    case Benchmark::Dot:     return time([&] { sum  = stream->dot(); });
    case Benchmark::Nstream: return time([&] { stream->nstream(); });
    default:
      std::cerr << "Unimplemented benchmark: " << i << "," <<  labels[i] << std::endl;
      abort();
    }
  };

  // Main loop
  for (size_t i = 0; i < num_benchmarks; ++i)
  {
    if (!run_benchmark(i)) continue;
    timings[i].reserve(num_times);
    for (unsigned int k = 0; k < num_times; k++) timings[i].push_back(dt(i));
  }

  // Compiler should use a move
  return timings;
}

// Generic run routine
// Runs the kernel(s) and prints output.
template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  // Formatting utilities:
  auto fmt_unit = [](double bytes, Unit unit = Unit::Mega) {
    switch(unit) {
    case Unit::Mega: return (mibibytes? pow(2.0, -20.0) : 1.0E-6) * bytes;
    case Unit::Giga: return (mibibytes? pow(2.0, -30.0) : 1.0E-9) * bytes;
    default: std::cerr << "Unimplemented!" << std::endl; abort();
    }
  };
  auto unit_label = [](Unit unit = Unit::Mega) {
    switch(unit) {
    case Unit::Mega: return mibibytes? "MiB" : "MB";
    case Unit::Giga: return mibibytes? "GiB" : "GB";
    default: std::cerr << "Unimplemented!" << std::endl; abort();
    }
  };
  auto fmt_bw = [&](size_t weight, double dt, Unit unit = Unit::Mega) {
    return fmt_unit((weight * sizeof(T) * ARRAY_SIZE)/dt, unit);
  };
  // Output formatting:
  auto fmt_csv_header = [] {
    std::cout
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << ((mibibytes) ? "max_mibytes_per_sec" : "max_mbytes_per_sec") << csv_separator
      << "min_runtime" << csv_separator
      << "max_runtime" << csv_separator
      << "avg_runtime" << std::endl;
  };
  auto fmt_csv = [](char const* function, size_t num_times, size_t num_elements,
                    size_t type_size, double bandwidth, double dt_min, double dt_max, double dt_avg) {
    std::cout << function << csv_separator
         << num_times << csv_separator
         << num_elements << csv_separator
         << type_size << csv_separator
         << bandwidth << csv_separator
         << dt_min << csv_separator
         << dt_max << csv_separator
         << dt_avg << std::endl;
  };
  auto fmt_cli = [](char const* function, double bandwidth, double dt_min, double dt_max, double dt_avg) {
    std::cout
      << std::left << std::setw(12) << function
      << std::left << std::setw(12) << setprecision(3) << bandwidth
      << std::left << std::setw(12) << setprecision(5) << dt_min
      << std::left << std::setw(12) << setprecision(5) << dt_max
      << std::left << std::setw(12) << setprecision(5) << dt_avg
      << std::endl;
  };
  auto fmt_result = [&](char const* function, size_t num_times, size_t num_elements,
                        size_t type_size, double bandwidth, double dt_min, double dt_max, double dt_avg) {
    if (!output_as_csv) return fmt_cli(function, bandwidth, dt_min, dt_max, dt_avg);
    fmt_csv(function, num_times, num_elements, type_size, bandwidth, dt_min, dt_max, dt_avg);
  };

  if (!output_as_csv)
  {
    std::cout << "Running ";
    switch(selection) {
    case Benchmark::All: std::cout << " All kernels "; break;
    case Benchmark::Classic: std::cout << " Classic kernels "; break;
    default: std::cout << "Running " << labels[(int)selection] << " ";
    }
    std::cout << num_times << " times" << std::endl;
    std::cout << "Number of elements: " << ARRAY_SIZE << std::endl;
    std::cout << "Precision: " << (sizeof(T) == sizeof(float)? "float" : "double") << std::endl;

    size_t nbytes = ARRAY_SIZE*sizeof(T);
    std::cout << setprecision(1) << fixed
	 << "Array size: " << fmt_unit(nbytes, Unit::Mega) << " " << unit_label(Unit::Mega)
	 << " (=" << fmt_unit(nbytes, Unit::Giga) << " " << unit_label(Unit::Giga) << ")" << std::endl;
    std::cout << "Total size: " << fmt_unit(3.0*nbytes, Unit::Mega) << " " << unit_label(Unit::Mega)
	 << " (=" << fmt_unit(3.0*nbytes, Unit::Giga) << " " << unit_label(Unit::Giga) << ")" << std::endl;
    std::cout.precision(ss);
  }

  auto stream = construct_stream<T>(ARRAY_SIZE, deviceIndex);
  auto initElapsedS = time([&] { stream->init_arrays(startA, startB, startC); });

  // Result of the Dot kernel, if used.
  T sum{};
  vector<vector<double>> timings = run_all<T>(stream, sum);

  // Create & read host vectors:
  vector<T> a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE);
  auto readElapsedS = time([&] { stream->read_arrays(a, b, c); });

  check_solution<T>(num_times, a, b, c, sum);
  auto initBWps = fmt_bw(3, initElapsedS);
  auto readBWps = fmt_bw(3, readElapsedS);

  if (output_as_csv)
  {
    fmt_csv_header();
    fmt_csv("Init", 1, ARRAY_SIZE, sizeof(T), initBWps, initElapsedS, initElapsedS, initElapsedS);
    fmt_csv("Read", 1, ARRAY_SIZE, sizeof(T), readBWps, readElapsedS, readElapsedS, readElapsedS);
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

    std::cout
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << ((mibibytes) ? "MiBytes/sec" : "MBytes/sec")
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << std::fixed;
  }

  for (int i = 0; i < num_benchmarks; ++i)
  {
    if (!run_benchmark(i)) continue;

    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

    // Display results
    fmt_result(labels[i], num_times, ARRAY_SIZE, sizeof(T), fmt_bw(weight[i], *minmax.first),
               *minmax.first, *minmax.second, average);
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

  for (int b = 0; b < num_benchmarks; ++b)
  {
    if (!run_benchmark(b)) continue;

    for (unsigned int i = 0; i < ntimes; i++)
    {
      switch((Benchmark)b) {
      case Benchmark::Copy:    goldC = goldA; break;
      case Benchmark::Mul:     goldB = scalar * goldC; break;
      case Benchmark::Add:     goldC = goldA + goldB; break;
      case Benchmark::Triad:   goldA = goldB + scalar * goldC; break;
      case Benchmark::Nstream: goldA += goldB + scalar * goldC; break;
      case Benchmark::Dot:     goldSum = goldA * goldB * ARRAY_SIZE; break;
      default: std::cerr << "Unimplemented Check: " << i << "," << labels[b] << std::endl; abort();
      }
    }
  }

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
    else if (!std::string("--print-names").compare(argv[i]))
    {
      std::cout << "Available benchmarks: ";
      print_labels(std::cout);
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else if (!std::string("--only").compare(argv[i]) || !std::string("-o").compare(argv[i]))
    {
      if (++i >= argc)
      {
	std::cerr << "Expected benchmark name after --only" << std::endl;
        exit(EXIT_FAILURE);
      }
      auto key = string(argv[i]);
      if (key == "Classic")
      {
        selection = Benchmark::Classic;
      }
      else if (key == "All")
      {
        selection = Benchmark::All;
      }
      else
      {
        auto p = find_if(labels.begin(), labels.end(), [&](char const* label) {
            return string(label) == key;
          });
        if (p == labels.end()) {
          std::cerr << "Unknown benchmark name \"" << argv[i] << "\" after --only" << std::endl;
          std::cerr << "Available benchmarks: All,Classic,";
          print_labels(std::cerr);
          std::cerr << std::endl;
	  std::exit(EXIT_FAILURE);
        }
        selection = (Benchmark)(distance(labels.begin(), p));
      }
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
      std::cout << "  -o  --only       NAME    Only run one benchmark (see --print-names)" << std::endl;
      std::cout << "      --print-names        Prints all available benchmark names" << std::endl;
      std::cout << "      --csv                Output as csv table" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << std::endl;
      std::exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}
