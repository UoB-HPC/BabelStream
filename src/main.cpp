
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

#ifdef ENABLE_CALIPER
#include <caliper/cali.h>
#include <caliper/cali-mpi.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#endif

// Default size of 2^25
intptr_t array_size = 33554432;
size_t num_times = 100;
size_t deviceIndex = 0;
bool use_float = false;
bool output_as_csv = false;
// Default unit of memory is MegaBytes (as per STREAM) 
Unit unit{Unit::Kind::MegaByte};
bool silence_errors = false;
std::string csv_separator = ",";

// Selected benchmarks to run: default is all 5 classic benchmarks.
BenchId selection = BenchId::Classic;

// Returns true if the benchmark needs to be run:
bool run_benchmark(Benchmark const& b) { return run_benchmark(selection, b); }

// Benchmark run order
// - Classic: runs each bench once in the order above, and repeats n times.
// - Isolated: runs each bench n times in isolation
enum class BenchOrder : int {Classic, Isolated};
BenchOrder order = BenchOrder::Classic;

template <typename T>
void run();

void parseArguments(int argc, char *argv[]);

int main(int argc, char *argv[])
{
#ifdef ENABLE_CALIPER
  	cali::ConfigManager calimgr;

     if (calimgr.error()){
      std::cerr << "caliper config error: " << calimgr.error_msg() << std::endl;
    }
    calimgr.start();

    cali_config_preset("CALI_LOG_VERBOSITY", "2");
    cali_config_preset("CALI_CALIPER_ATTRIBUTE_DEFAULT_SCOPE", "process");

    adiak::init(nullptr);
    adiak::collect_all();

    cali_config_set("CALI_CALIPER_ATTRIBUTE_DEFAULT_SCOPE", "process");

	adiak::value("BABELSTREAM version", "4.0");
	adiak::value("num_times", num_times);
	adiak::value("elements", ARRAY_SIZE);

	CALI_MARK_FUNCTION_BEGIN;
#endif  

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

#ifdef ENABLE_CALIPER
    adiak::fini();
    calimgr.flush();
#endif
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

  // Run a particular benchmark
  auto run = [&](Benchmark const& b)
  {
    switch(b.id) {
    case BenchId::Copy:    return stream->copy();
    case BenchId::Mul:     return stream->mul();
    case BenchId::Add:     return stream->add();
    case BenchId::Triad:   return stream->triad();
    case BenchId::Dot:     sum = stream->dot(); return;
    case BenchId::Nstream: return stream->nstream();
    default:
      std::cerr << "Unimplemented benchmark: " << b.label << std::endl;
      abort();
    }
  };

  // Time a particular benchmark:
  auto dt = [&](Benchmark const& b) { return time([&] { run(b); }); };

  // Reserve timings:
  for (size_t i = 0; i < num_benchmarks; ++i) {
    if (!run_benchmark(bench[i])) continue;
    timings[i].reserve(num_times);
  }

  switch(order) {
  // Classic runs each benchmark once in the order specifies in the "bench" array above,
  // and then repeats num_times:
  case BenchOrder::Classic: {
    for (size_t k = 0; k < num_times; k++) {
      for (size_t i = 0; i < num_benchmarks; ++i) {
	if (!run_benchmark(bench[i])) continue;
#ifdef ENABLE_CALIPER
    CALI_MARK_BEGIN(bench[i].label);
#endif 
	timings[i].push_back(dt(bench[i]));
#ifdef ENABLE_CALIPER
    CALI_MARK_END(bench[i].label);
#endif 
      }
    }
    break;
  }
  // Isolated runs each benchmark num_times, before proceeding to run the next benchmark:
  case BenchOrder::Isolated: {
    for (size_t i = 0; i < num_benchmarks; ++i) {
      if (!run_benchmark(bench[i])) continue;
      auto t = time([&] { for (size_t k = 0; k < num_times; k++) run(bench[i]); });
      timings[i].resize(num_times, t / (double)num_times);
    }
    break;
  }
  default:
    std::cerr << "Unimplemented order" << std::endl;
    abort();
  }

  // Compiler should use a move
  return timings;
}

template <typename T>
void check_solution(const size_t ntimes, T const* a, T const* b, T const* c, T sum);

// Generic run routine
// Runs the kernel(s) and prints output.
template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  // Formatting utilities:
  auto fmt_bw = [&](size_t weight, double dt) {
    return unit.fmt((weight * sizeof(T) * array_size)/dt);
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
    std::cout << num_times << " times in ";
    switch (order) {
    case BenchOrder::Classic: std::cout << " Classic"; break;
    case BenchOrder::Isolated: std::cout << " Isolated"; break;
    default: std::cerr << "Error: Unknown order" << std::endl; abort();
    };
    std::cout << " order " << std::endl;
    std::cout << "Number of elements: " << array_size << std::endl;
    std::cout << "Precision: " << (sizeof(T) == sizeof(float)? "float" : "double") << std::endl;

    size_t nbytes = array_size * sizeof(T);
    std::cout << std::setprecision(1) << std::fixed
	      << "Array size: " << unit.fmt(nbytes) << " " << unit.str() << std::endl;
    std::cout << "Total size: " << unit.fmt(3.0*nbytes) << " " << unit.str() << std::endl;
    std::cout.precision(ss);
  }

  std::unique_ptr<Stream<T>> stream
    = make_stream<T>(selection, array_size, deviceIndex, startA, startB, startC);
  
#ifdef ENABLE_CALIPER
    CALI_MARK_BEGIN("init_arrays");
#endif

  auto initElapsedS = time([&] { stream->init_arrays(startA, startB, startC); });

#ifdef ENABLE_CALIPER
    CALI_MARK_END("init_arrays");
#endif

  // Result of the Dot kernel, if used.
  T sum{};
  std::vector<std::vector<double>> timings = run_all<T>(stream, sum);

  // Create & read host vectors:
  T const* a;
  T const* b;
  T const* c;
  stream->get_arrays(a, b, c);

  check_solution<T>(num_times, a, b, c, sum);

  if (output_as_csv)
  {
    fmt_csv_header();
  }
  else
  {
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
    fmt_result(bench[i].label, num_times, array_size, sizeof(T),
	       fmt_bw(bench[i].weight, *minmax.first), *minmax.first, *minmax.second, average);
  }
}

template <typename T>
void check_solution(const size_t num_times, T const* a, T const* b, T const* c, T sum) {
  // Generate correct solution
  T goldA = startA;
  T goldB = startB;
  T goldC = startC;
  T goldS = T(0.);

  const T scalar = startScalar;

  // Updates output due to running each benchmark:
  auto run = [&](int b) {
    switch(bench[b].id) {
    case BenchId::Copy:    goldC = goldA; break;
    case BenchId::Mul:     goldB = scalar * goldC; break;
    case BenchId::Add:     goldC = goldA + goldB; break;
    case BenchId::Triad:   goldA = goldB + scalar * goldC; break;
    case BenchId::Nstream: goldA += goldB + scalar * goldC; break;
    case BenchId::Dot:     goldS = goldA * goldB * T(array_size); break; // This calculates the answer exactly
    default:
    std::cerr << "Unimplemented Check: " << bench[b].label << std::endl;
    abort();
    }
  };

  switch(order) {
  // Classic runs each benchmark once in the order specifies in the "bench" array above,
  // and then repeats num_times:
  case BenchOrder::Classic: {
    for (size_t k = 0; k < num_times; k++) {
      for (size_t i = 0; i < num_benchmarks; ++i) {
	      if (!run_benchmark(bench[i])) continue;
	      run(i);
      }
    }
    break;
  }
  // Isolated runs each benchmark num_times, before proceeding to run the next benchmark:
  case BenchOrder::Isolated: {
    for (size_t i = 0; i < num_benchmarks; ++i) {
      if (!run_benchmark(bench[i])) continue;
      for (size_t k = 0; k < num_times; k++) run(i);
    }
    break;
  }
  default:
    std::cerr << "Unimplemented order" << std::endl;
    abort();
  }

  // Error relative tolerance check - a higher tolerance is used for reductions.
  size_t failed = 0;
  T max_rel = std::numeric_limits<T>::epsilon() * T(100.0);
  T max_rel_dot = std::numeric_limits<T>::epsilon() * T(10000000.0);
  auto check = [&](const char* name, T is, T should, T mrel, size_t i = size_t(-1)) {
    // Relative difference:
    T diff = std::abs(is - should);
    T abs_is = std::abs(is);
    T abs_sh = std::abs(should);
    T largest = std::max(abs_is, abs_sh);
    T same = diff <= largest * mrel;
    if (!same || std::isnan(is)) {
      ++failed;
      if (failed > 10) return;
      std::cerr << "FAILED validation of " << name;
      if (i != size_t(-1)) std::cerr << "[" << i << "]";
      std::cerr << ": " << is << " (is) != " << should
		<< " (should)" << ", diff=" << diff << " > "
		<< largest * mrel << " (largest=" << largest
		<< ", max_rel=" << mrel << ")" << std::endl;
    }
  };

  // Sum
  for (size_t i = 0; i < num_benchmarks; ++i) {
    if (bench[i].id != BenchId::Dot) continue;
    if (run_benchmark(bench[i]))
      check("sum", sum, goldS, max_rel_dot);
    break;
  }

  // Calculate the L^infty-norm relative error
  for (size_t i = 0; i < array_size; ++i) {
    check("a", a[i], goldA, max_rel, i);
    check("b", b[i], goldB, max_rel, i);
    check("c", c[i], goldC, max_rel, i);
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
  auto parseInt = [](const char *str, intptr_t *output) {
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
      if (++i >= argc || !parseInt(argv[i], &array_size) || array_size <= 0)
      {
        std::cerr << "Invalid array size." << std::endl;
        std::exit(EXIT_FAILURE);
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
      std::exit(EXIT_SUCCESS);
    }
    else if (!std::string("--only").compare(argv[i]) || !std::string("-o").compare(argv[i]))
    {
      if (++i >= argc)
      {
        std::cerr << "Expected benchmark name after --only" << std::endl;
        std::exit(EXIT_FAILURE);
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
    else if (!std::string("--order").compare(argv[i]))
    {
      if (++i >= argc)
      {
       std::cerr << "Expected benchmark order after --order. Options: \"Classic\" (default), \"Isolated\"."
		  << std::endl;
        exit(EXIT_FAILURE);
      }
      auto key = std::string(argv[i]);
      if (key == "Isolated")
      {
        order = BenchOrder::Isolated;
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
      std::cout << "      --order              Benchmark run order: \"Classic\" (default) or \"Isolated\"." << std::endl;
      std::cout << "      --csv                Output as csv table" << std::endl;
      std::cout << "      --megabytes          Use MB=10^6 for bandwidth calculation (default)" << std::endl;
      std::cout << "      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)" << std::endl;
      std::cout << "      --gibibytes          Use GiB=2^30 for bandwidth calculation (default MB=10^6)" << std::endl;
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
