// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <algorithm>
#include <array>
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
#include "Unit.h"

// Default size of 2^25
int ARRAY_SIZE = 33554432;
size_t num_times = 100;
size_t deviceIndex = 0;
bool use_float = false;
bool output_as_csv = false;
Unit unit{Unit::Kind::MegaByte};
bool silence_errors = false;
std::string csv_separator = ",";

// Benchmark Identifier: identifies individual & groups of benchmarks:
// - Classic: 5 classic kernels: Copy, Mul, Add, Triad, Dot.
// - All: all kernels.
// - Individual kernels only.
enum class BenchId : int {Copy, Mul, Add, Triad, Nstream, Dot, Classic, All};

struct Benchmark {
  BenchId id;
  char const* label;
  // Weights data moved by benchmark & therefore achieved BW:
  // bytes = weight * sizeof(T) * ARRAY_SIZE -> bw = bytes / dur
  size_t weight;
  // Is it one of: Copy, Mul, Add, Triad, Dot?
  bool classic = false;
};

// Benchmarks in the order in which - if present - should be run for validation purposes:
constexpr size_t num_benchmarks = 6;
std::array<Benchmark, num_benchmarks> bench = {
  Benchmark { .id = BenchId::Copy,    .label = "Copy",    .weight = 2, .classic = true  },
  Benchmark { .id = BenchId::Mul,     .label = "Mul",     .weight = 2, .classic = true  },
  Benchmark { .id = BenchId::Add,     .label = "Add",     .weight = 3, .classic = true  },
  Benchmark { .id = BenchId::Triad,   .label = "Triad",   .weight = 3, .classic = true  },
  Benchmark { .id = BenchId::Dot,     .label = "Dot",     .weight = 2, .classic = true  },
  Benchmark { .id = BenchId::Nstream, .label = "Nstream", .weight = 4, .classic = false }
};

// Selected benchmarks to run: default is all 5 classic benchmarks.
BenchId selection = BenchId::Classic;

// Returns true if the benchmark needs to be run:
bool run_benchmark(Benchmark const& b) {
  if (selection == BenchId::All)                  return true;
  if (selection == BenchId::Classic && b.classic) return true;
  return selection == b.id;
}

template <typename T>
void run();

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

  return EXIT_SUCCESS;
}

// Returns duration of executing function f:
template <typename F>
double time(F&& f) {
  using clk_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double>;
  auto start = clk_t::now();
  f();
  return dur_t(clk_t::now() - start).count();
}

// Run specified kernels
template <typename T>
std::vector<std::vector<double>> run_all(std::unique_ptr<Stream<T>>& stream, T& sum)
{
  // Times for each measured benchmark:
  std::vector<std::vector<double>> timings(num_benchmarks);

  // Time a particular benchmark:
  auto dt = [&](Benchmark const& b)
  {
    switch(b.id) {
    case BenchId::Copy:    return time([&] { stream->copy(); });
    case BenchId::Mul:     return time([&] { stream->mul(); });
    case BenchId::Add:     return time([&] { stream->add(); });
    case BenchId::Triad:   return time([&] { stream->triad(); });
    case BenchId::Dot:     return time([&] { sum = stream->dot(); });
    case BenchId::Nstream: return time([&] { stream->nstream(); });
    default:
      std::cerr << "Unimplemented benchmark: " << b.label << std::endl;
      abort();
    }
  };

  // Main loop
  for (size_t i = 0; i < num_benchmarks; ++i)
  {
    if (!run_benchmark(bench[i])) continue;
    timings[i].reserve(num_times);
    for (size_t k = 0; k < num_times; k++) timings[i].push_back(dt(bench[i]));
  }

  // Compiler should use a move
  return timings;
}

template <typename T>
void check_solution(const size_t ntimes, std::vector<T>& a, std::vector<T>& b, std::vector<T>& c,
		    T& sum);

// Generic run routine
// Runs the kernel(s) and prints output.
template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  // Formatting utilities:
  auto fmt_bw = [&](size_t weight, double dt) {
    return unit.fmt((weight * sizeof(T) * ARRAY_SIZE)/dt);
  };
  auto fmt_csv_header = [] {
    std::cout
      << "function" << csv_separator
      << "num_times" << csv_separator
      << "n_elements" << csv_separator
      << "sizeof" << csv_separator
      << "max_" << unit.str() << "_per_sec" << csv_separator
      << "min_runtime" << csv_separator
      << "max_runtime" << csv_separator
      << "avg_runtime" << std::endl;
  };
  auto fmt_csv = [](char const* function, size_t num_times, size_t num_elements,
                    size_t type_size, double bandwidth,
		    double dt_min, double dt_max, double dt_avg) {
    std::cout << function << csv_separator
         << num_times << csv_separator
         << num_elements << csv_separator
         << type_size << csv_separator
         << bandwidth << csv_separator
         << dt_min << csv_separator
         << dt_max << csv_separator
         << dt_avg << std::endl;
  };
  auto fmt_cli = [](char const* function, double bandwidth,
		    double dt_min, double dt_max, double dt_avg) {
    std::cout
      << std::left << std::setw(12) << function
      << std::left << std::setw(12) << std::setprecision(3) << bandwidth
      << std::left << std::setw(12) << std::setprecision(5) << dt_min
      << std::left << std::setw(12) << std::setprecision(5) << dt_max
      << std::left << std::setw(12) << std::setprecision(5) << dt_avg
      << std::endl;
  };
  auto fmt_result = [&](char const* function, size_t num_times, size_t num_elements,
                        size_t type_size, double bandwidth,
			double dt_min, double dt_max, double dt_avg) {
    if (!output_as_csv) return fmt_cli(function, bandwidth, dt_min, dt_max, dt_avg);
    fmt_csv(function, num_times, num_elements, type_size, bandwidth, dt_min, dt_max, dt_avg);
  };

  if (!output_as_csv)
  {
    std::cout << "Running ";
    switch(selection) {
    case BenchId::All: std::cout << " All kernels "; break;
    case BenchId::Classic: std::cout << " Classic kernels "; break;
    default:
      std::cout << "Running ";
      for (size_t i = 0; i < num_benchmarks; ++i) {
	if (selection == bench[i].id) {
	  std::cout << bench[i].label;
	  break;
	}
      }
      std::cout << " ";
    }
    std::cout << num_times << " times" << std::endl;
    std::cout << "Number of elements: " << ARRAY_SIZE << std::endl;
    std::cout << "Precision: " << (sizeof(T) == sizeof(float)? "float" : "double") << std::endl;

    size_t nbytes = ARRAY_SIZE * sizeof(T);
    std::cout << std::setprecision(1) << std::fixed
	      << "Array size: " << unit.fmt(nbytes) << " " << unit.str() << std::endl;
    std::cout << "Total size: " << unit.fmt(3.0*nbytes) << " " << unit.str() << std::endl;
    std::cout.precision(ss);
  }

  std::unique_ptr<Stream<T>> stream = make_stream<T>(ARRAY_SIZE, deviceIndex);
  auto initElapsedS = time([&] { stream->init_arrays(startA, startB, startC); });

  // Result of the Dot kernel, if used.
  T sum{};
  std::vector<std::vector<double>> timings = run_all<T>(stream, sum);

  // Create & read host vectors:
  std::vector<T> a(ARRAY_SIZE), b(ARRAY_SIZE), c(ARRAY_SIZE);
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
      << initElapsedS << " s (=" << initBWps << " " << unit.str() << "/s" << ")" << std::endl;
    std::cout << "Read: "
      << std::setw(7)
      << readElapsedS << " s (=" << readBWps << " " << unit.str() << "/s" << ")" << std::endl;

    std::cout
      << std::left << std::setw(12) << "Function"
      << std::left << std::setw(12) << (std::string(unit.str()) + "/s")
      << std::left << std::setw(12) << "Min (sec)"
      << std::left << std::setw(12) << "Max"
      << std::left << std::setw(12) << "Average"
      << std::endl
      << std::fixed;
  }

  for (size_t i = 0; i < num_benchmarks; ++i)
  {
    if (!run_benchmark(bench[i])) continue;

    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0)
      / (double)(num_times - 1);

    // Display results
    fmt_result(bench[i].label, num_times, ARRAY_SIZE, sizeof(T),
	       fmt_bw(bench[i].weight, *minmax.first), *minmax.first, *minmax.second, average);
  }
}

template <typename T>
void check_solution(const size_t num_times,
		    std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, T& sum)
{
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;
  T goldS = T(0.);

  const T scalar = startScalar;

  for (size_t b = 0; b < num_benchmarks; ++b)
  {
    if (!run_benchmark(bench[b])) continue;

    for (size_t i = 0; i < num_times; i++)
    {
      switch(bench[b].id) {
      case BenchId::Copy:    goldC = goldA; break;
      case BenchId::Mul:     goldB = scalar * goldC; break;
      case BenchId::Add:     goldC = goldA + goldB; break;
      case BenchId::Triad:   goldA = goldB + scalar * goldC; break;
      case BenchId::Nstream: goldA += goldB + scalar * goldC; break;
      case BenchId::Dot:     goldS = goldA * goldB * T(ARRAY_SIZE); break;
      default:
	std::cerr << "Unimplemented Check: " << bench[b].label << std::endl;
	abort();
      }
    }
  }

  // Error relative tolerance check
  size_t failed = 0;
  T epsi = std::numeric_limits<T>::epsilon() * T(100000.0);
  auto check = [&](const char* name, T is, T should, T e, size_t i = size_t(-1)) {
    if (e > epsi) {
      ++failed;
      if (failed > 10) return;
      std::cerr << "FAILED validation of " << name;
      if (i != size_t(-1)) std::cerr << "[" << i << "]";
      std::cerr << ": " << is << " != " << should
		<< ", relative error=" << e << " > " << epsi << std::endl;
    }
  };

  // Sum
  T eS = std::fabs(sum - goldS) / std::fabs(goldS);
  for (size_t i = 0; i < num_benchmarks; ++i) {
    if (bench[i].id != BenchId::Dot) continue;
    if (run_benchmark(bench[i]))
      check("sum", sum,  goldS, eS);
    break;
  }

  // Calculate the L^infty-norm relative error
  for (size_t i = 0; i < a.size(); ++i) {
    T vA = a[i], vB = b[i], vC = c[i];
    T eA = std::fabs(vA - goldA) / std::fabs(goldA);
    T eB = std::fabs(vB - goldB) / std::fabs(goldB);
    T eC = std::fabs(vC - goldC) / std::fabs(goldC);

    check("a", a[i], goldA, eA, i);
    check("b", b[i], goldB, eB, i);
    check("c", c[i], goldC, eC, i);
  }

  if (failed > 0 && !silence_errors)
    std::exit(EXIT_FAILURE);
}

void parseArguments(int argc, char *argv[])
{
  auto parseUInt =[](const char *str, size_t *output) {
    char *next;
    *output = strtoull(str, &next, 10);
    return !strlen(next);
  };
  auto parseInt =[](const char *str, intptr_t *output) {
    char *next;
    *output = strtoll(str, &next, 10);
    return !strlen(next);
  };

  // Prints all available benchmark labels:
  auto print_labels = [&](auto& os) {
    for (size_t i = 0; i < num_benchmarks; ++i) {
      os << bench[i].label;
      if (i != (num_benchmarks - 1)) os << ",";
    }
  };

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
      intptr_t array_size;
      if (++i >= argc || !parseInt(argv[i], &array_size) || array_size <= 0)
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
      ARRAY_SIZE = array_size;
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
      auto key = std::string(argv[i]);
      if (key == "Classic")
      {
        selection = BenchId::Classic;
      }
      else if (key == "All")
      {
        selection = BenchId::All;
      }
      else
      {
        auto p = std::find_if(bench.begin(), bench.end(), [&](Benchmark const& b) {
	  return std::string(b.label) == key;
        });
        if (p == bench.end()) {
          std::cerr << "Unknown benchmark name \"" << argv[i] << "\" after --only" << std::endl;
          std::cerr << "Available benchmarks: All, Classic,";
          print_labels(std::cerr);
          std::cerr << std::endl;
	  std::exit(EXIT_FAILURE);
        }
        selection = p->id;
      }
    }
    else if (!std::string("--csv").compare(argv[i]))
    {
      output_as_csv = true;
    }
    else if (!std::string("--mibibytes").compare(argv[i]))
    {
      unit = Unit(Unit::Kind::MibiByte);
    }
    else if (!std::string("--megabytes").compare(argv[i]))
    {
      unit = Unit(Unit::Kind::MegaByte);
    }
    else if (!std::string("--gibibytes").compare(argv[i]))
    {
      unit = Unit(Unit::Kind::GibiByte);
    }
    else if (!std::string("--gigabytes").compare(argv[i]))
    {
      unit = Unit(Unit::Kind::GigaByte);
    }
    else if (!std::string("--tebibytes").compare(argv[i]))
    {
      unit = Unit(Unit::Kind::TebiByte);
    }
    else if (!std::string("--terabytes").compare(argv[i]))
    {
      unit = Unit(Unit::Kind::TeraByte);
    }
    else if (!std::string("--silence-errors").compare(argv[i]))
    {
      silence_errors = true;
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
      std::cout << "      --megabytes          Use MB=10^6 for bandwidth calculation (default)" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --gigibytes          Use GiB=2^30 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --gigabytes          Use GB=10^9 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --tebibytes          Use TiB=2^40 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --terabytes          Use TB=10^12 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --silence-errors     Ignores validation errors." << std::endl;
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
