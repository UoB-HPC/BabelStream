#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>

// Array values
#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

// Benchmark Identifier: identifies individual & groups of benchmarks:
// - Classic: 5 classic kernels: Copy, Mul, Add, Triad, Dot.
// - All: all kernels.
// - Individual kernels only.
enum class BenchId : int {Copy, Mul, Add, Triad, Nstream, Dot, Classic, All};

struct Benchmark {
  BenchId id;
  char const* label;
  // Weight counts data elements of original arrays moved each loop iteration - used to calculate achieved BW:
  // bytes = weight * sizeof(T) * ARRAY_SIZE -> bw = bytes / dur
  size_t weight;
  // Is it one of: Copy, Mul, Add, Triad, Dot?
  bool classic = false;
};

// Benchmarks in the order in which - if present - should be run for validation purposes:
constexpr size_t num_benchmarks = 6;
constexpr std::array<Benchmark, num_benchmarks> bench = {
  Benchmark { .id = BenchId::Copy,    .label = "Copy",    .weight = 2, .classic = true  },
  Benchmark { .id = BenchId::Mul,     .label = "Mul",     .weight = 2, .classic = true  },
  Benchmark { .id = BenchId::Add,     .label = "Add",     .weight = 3, .classic = true  },
  Benchmark { .id = BenchId::Triad,   .label = "Triad",   .weight = 3, .classic = true  },
  Benchmark { .id = BenchId::Dot,     .label = "Dot",     .weight = 2, .classic = true  },
  Benchmark { .id = BenchId::Nstream, .label = "Nstream", .weight = 4, .classic = false }
};

// Which buffers are needed by each benchmark
inline bool needs_buffer(BenchId id, char n) {
  auto in = [n](std::initializer_list<char> values) {
    return std::find(values.begin(), values.end(), n) != values.end();
  };
  switch(id) {
  case BenchId::All:     return in({'a','b','c'});       
  case BenchId::Classic: return in({'a','b','c'});   
  case BenchId::Copy:    return in({'a','c'});
  case BenchId::Mul:	 return in({'b','c'});
  case BenchId::Add:	 return in({'a','b','c'});
  case BenchId::Triad:   return in({'a','b','c'});
  case BenchId::Dot:	 return in({'a','b'});
  case BenchId::Nstream: return in({'a','b','c'});  
  default:
    std::cerr << "Unknown benchmark" << std::endl;
    abort();
  }
}

// Returns true if the benchmark needs to be run:
inline bool run_benchmark(BenchId selection, Benchmark const& b) {
  if (selection == BenchId::All)                  return true;
  if (selection == BenchId::Classic && b.classic) return true;
  return selection == b.id;
}
