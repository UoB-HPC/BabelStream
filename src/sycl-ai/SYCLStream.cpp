
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream.h"

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <limits>

using namespace sycl;

namespace
{
size_t round_up(size_t value, size_t multiple)
{
  if (multiple == 0)
    return value;
  const size_t rem = value % multiple;
  return rem == 0 ? value : value + (multiple - rem);
}

size_t env_to_size_t(const char *name, size_t fallback)
{
  const char *v = std::getenv(name);
  if (v == nullptr)
    return fallback;

  char *end = nullptr;
  const unsigned long parsed = std::strtoul(v, &end, 10);
  if (end == v || *end != '\0' || parsed == 0)
    return fallback;

  return static_cast<size_t>(parsed);
}

bool env_to_bool(const char *name, bool fallback)
{
  const char *v = std::getenv(name);
  if (v == nullptr)
    return fallback;

  if (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE")
    return true;
  if (std::string(v) == "0" || std::string(v) == "false" || std::string(v) == "FALSE")
    return false;

  return fallback;
}

bool env_is_set(const char *name)
{
  return std::getenv(name) != nullptr;
}
}

// Cache list of devices
bool cached = false;
std::vector<device> devices;
void getDeviceList(void);

template <class T>
SYCLStream<T>::SYCLStream(const int ARRAY_SIZE, const int device_index)
{
  if (!cached)
    getDeviceList();

  array_size = ARRAY_SIZE;

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device dev = devices[device_index];

  // Check device can support FP64 if needed
  if (sizeof(T) == sizeof(double))
  {
    if (dev.get_info<info::device::double_fp_config>().size() == 0) {
      throw std::runtime_error("Device does not support double precision, please use --float");
    }
  }

  const size_t max_wg = dev.get_info<info::device::max_work_group_size>();

  // Determine sensible kernel NDRange configuration, with optional env overrides.
  // Useful for tuning on a specific Intel GPU without recompiling.
  const bool is_gpu = dev.is_gpu();
  const bool is_intel_gpu = is_gpu && dev.get_info<info::device::vendor>().find("Intel") != std::string::npos;
  const size_t default_stream_wg = is_intel_gpu ? size_t{1024} : (is_gpu ? size_t{256} : size_t{64});
  stream_wgsize = std::min(max_wg, env_to_size_t("BABELSTREAM_SYCL_WG_SIZE", default_stream_wg));
  if (stream_wgsize == 0)
    stream_wgsize = std::min<size_t>(max_wg, 1);

  copy_wgsize = std::min(max_wg, env_to_size_t("BABELSTREAM_SYCL_COPY_WG_SIZE", stream_wgsize));
  mul_wgsize = std::min(max_wg, env_to_size_t("BABELSTREAM_SYCL_MUL_WG_SIZE", stream_wgsize));
  add_wgsize = std::min(max_wg, env_to_size_t("BABELSTREAM_SYCL_ADD_WG_SIZE", stream_wgsize));
  triad_wgsize = std::min(max_wg, env_to_size_t("BABELSTREAM_SYCL_TRIAD_WG_SIZE", stream_wgsize));
  nstream_wgsize = std::min(max_wg, env_to_size_t("BABELSTREAM_SYCL_NSTREAM_WG_SIZE", stream_wgsize));

  if (copy_wgsize == 0) copy_wgsize = std::min<size_t>(max_wg, 1);
  if (mul_wgsize == 0) mul_wgsize = std::min<size_t>(max_wg, 1);
  if (add_wgsize == 0) add_wgsize = std::min<size_t>(max_wg, 1);
  if (triad_wgsize == 0) triad_wgsize = std::min<size_t>(max_wg, 1);
  if (nstream_wgsize == 0) nstream_wgsize = std::min<size_t>(max_wg, 1);

  const size_t groups_per_cu_default = is_intel_gpu ? size_t{8} : (is_gpu ? size_t{8} : size_t{1});
  const size_t groups_per_cu = env_to_size_t("BABELSTREAM_SYCL_DOT_GROUPS_PER_CU", groups_per_cu_default);
  use_manual_dot_reduction = env_to_bool("BABELSTREAM_SYCL_DOT_MANUAL_REDUCTION", is_intel_gpu);

  dot_wgsize = std::min(max_wg, env_to_size_t("BABELSTREAM_SYCL_DOT_WG_SIZE", stream_wgsize));
  if (dot_wgsize == 0)
    dot_wgsize = std::min<size_t>(max_wg, 1);

  dot_unroll = env_to_size_t("BABELSTREAM_SYCL_DOT_UNROLL", is_intel_gpu ? size_t{4} : size_t{1});
  if (dot_unroll == 0)
    dot_unroll = 1;

  if (dev.is_cpu())
  {
    dot_num_groups = std::max<size_t>(1, dev.get_info<info::device::max_compute_units>() * groups_per_cu);
  }
  else
  {
    dot_num_groups = std::max<size_t>(1, dev.get_info<info::device::max_compute_units>() * groups_per_cu);
  }

  const size_t compute_units = std::max<size_t>(1, dev.get_info<info::device::max_compute_units>());
  dot_sum_capacity = std::max<size_t>(dot_num_groups, compute_units * 32);

  queue = std::make_unique<sycl::queue>(dev, sycl::async_handler{[&](sycl::exception_list l)
  {
    bool error = false;
    for(auto e: l)
    {
      try
      {
        std::rethrow_exception(e);
      }
      catch (sycl::exception e)
      {
        std::cout << e.what();
        error = true;
      }
    }
    if(error)
    {
      throw std::runtime_error("SYCL errors detected");
    }
  }});

  d_a = sycl::malloc_device<T>(array_size, *queue);
  d_b = sycl::malloc_device<T>(array_size, *queue);
  d_c = sycl::malloc_device<T>(array_size, *queue);
  d_sum = sycl::malloc_device<T>(dot_sum_capacity, *queue);
  d_dot = sycl::malloc_device<T>(1, *queue);
  h_sum = new T[dot_sum_capacity];

  const bool user_pinned_dot =
      env_is_set("BABELSTREAM_SYCL_DOT_WG_SIZE") ||
      env_is_set("BABELSTREAM_SYCL_DOT_GROUPS_PER_CU") ||
      env_is_set("BABELSTREAM_SYCL_DOT_UNROLL") ||
      env_is_set("BABELSTREAM_SYCL_DOT_MANUAL_REDUCTION");
  const bool user_pinned_stream =
      env_is_set("BABELSTREAM_SYCL_WG_SIZE") ||
      env_is_set("BABELSTREAM_SYCL_COPY_WG_SIZE") ||
      env_is_set("BABELSTREAM_SYCL_MUL_WG_SIZE") ||
      env_is_set("BABELSTREAM_SYCL_ADD_WG_SIZE") ||
      env_is_set("BABELSTREAM_SYCL_TRIAD_WG_SIZE") ||
      env_is_set("BABELSTREAM_SYCL_NSTREAM_WG_SIZE");
  const bool enable_stream_autotune = env_to_bool("BABELSTREAM_SYCL_STREAM_AUTOTUNE", false);
  const bool enable_dot_autotune = env_to_bool("BABELSTREAM_SYCL_DOT_AUTOTUNE", is_intel_gpu && !user_pinned_dot);

  if (enable_stream_autotune)
  {
    autotune_stream_kernels(is_intel_gpu, max_wg);
  }

  if (enable_dot_autotune)
  {
    autotune_dot(is_intel_gpu, max_wg);
  }

  // Print out final device and tuning information
  std::cout << "Using SYCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
  std::cout << "Stream kernel config: copy=" << copy_wgsize
            << " mul=" << mul_wgsize
            << " add=" << add_wgsize
            << " triad=" << triad_wgsize
            << " nstream=" << nstream_wgsize << std::endl;
  std::cout << "Reduction kernel config: " << dot_num_groups << " groups of size " << dot_wgsize << std::endl;
  std::cout << "Dot unroll factor: " << dot_unroll << std::endl;
  std::cout << "Dot reduction mode: " << (use_manual_dot_reduction ? "manual" : "native") << std::endl;
  if (enable_stream_autotune)
    std::cout << "Stream autotune: enabled" << std::endl;
  else
    std::cout << "Stream autotune: disabled" << std::endl;
  if (enable_dot_autotune)
    std::cout << "Dot autotune: enabled" << std::endl;
  else
    std::cout << "Dot autotune: disabled" << std::endl;
}

template <class T>
void SYCLStream<T>::autotune_stream_kernels(bool is_intel_gpu, size_t max_wg)
{
  queue->memset(d_a, 0, array_size * sizeof(T));
  queue->memset(d_b, 0, array_size * sizeof(T));
  queue->memset(d_c, 0, array_size * sizeof(T));
  queue->wait();

  std::vector<size_t> wg_candidates;
  for (size_t wg : {size_t{128}, size_t{256}, size_t{512}, size_t{1024}})
  {
    if (wg <= max_wg)
      wg_candidates.push_back(wg);
  }
  if (wg_candidates.empty())
    wg_candidates.push_back(std::min<size_t>(max_wg, stream_wgsize));

  const int trials = static_cast<int>(env_to_size_t("BABELSTREAM_SYCL_STREAM_AUTOTUNE_TRIALS", is_intel_gpu ? 3 : 2));

  auto tune_one = [&](size_t &wg, const auto &kernel_call)
  {
    double best_time = std::numeric_limits<double>::max();
    size_t best_wg = wg;

    for (auto cand : wg_candidates)
    {
      wg = cand;
      kernel_call();

      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < trials; i++)
        kernel_call();
      auto t2 = std::chrono::high_resolution_clock::now();

      const double avg = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() / trials;
      if (avg < best_time)
      {
        best_time = avg;
        best_wg = cand;
      }
    }

    wg = best_wg;
  };

  tune_one(copy_wgsize, [this]() { this->copy(); });
  tune_one(mul_wgsize, [this]() { this->mul(); });
  tune_one(add_wgsize, [this]() { this->add(); });
  tune_one(triad_wgsize, [this]() { this->triad(); });
  tune_one(nstream_wgsize, [this]() { this->nstream(); });
}

template <class T>
void SYCLStream<T>::autotune_dot(bool is_intel_gpu, size_t max_wg)
{
  // Initialize arrays to avoid undefined data effects during tuning.
  this->init_arrays(static_cast<T>(1), static_cast<T>(2), static_cast<T>(0));

  const auto dev = queue->get_device();
  const size_t compute_units = std::max<size_t>(1, dev.get_info<info::device::max_compute_units>());
  const int trials = static_cast<int>(env_to_size_t("BABELSTREAM_SYCL_DOT_AUTOTUNE_TRIALS", 2));

  std::vector<size_t> wg_candidates;
  for (size_t wg : {size_t{256}, size_t{512}, size_t{1024}})
  {
    if (wg <= max_wg)
      wg_candidates.push_back(wg);
  }
  if (wg_candidates.empty())
    wg_candidates.push_back(std::min<size_t>(max_wg, stream_wgsize));

  const std::vector<size_t> gpc_candidates = is_intel_gpu
      ? std::vector<size_t>{4, 6, 8, 12}
      : std::vector<size_t>{2, 4, 8};
  const std::vector<size_t> unroll_candidates = {1, 2, 4};

  bool best_manual = use_manual_dot_reduction;
  size_t best_wg = dot_wgsize;
  size_t best_groups = dot_num_groups;
  size_t best_unroll = dot_unroll;
  double best_time = std::numeric_limits<double>::max();

  auto bench_once = [this, trials](bool manual)->double
  {
    use_manual_dot_reduction = manual;

    volatile T sink = this->dot();
    (void)sink;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++)
    {
      sink = this->dot();
      (void)sink;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() / trials;
  };

  // Candidate 1: native reduction
  {
    const double t = bench_once(false);
    if (t < best_time)
    {
      best_time = t;
      best_manual = false;
      best_wg = dot_wgsize;
      best_groups = dot_num_groups;
      best_unroll = dot_unroll;
    }
  }

  // Candidate set 2: manual reduction tuning
  for (auto wg : wg_candidates)
  {
    for (auto gpc : gpc_candidates)
    {
      for (auto unroll : unroll_candidates)
      {
        dot_wgsize = wg;
        dot_num_groups = std::min(dot_sum_capacity, std::max<size_t>(1, compute_units * gpc));
        dot_unroll = unroll;
        const double t = bench_once(true);
        if (t < best_time)
        {
          best_time = t;
          best_manual = true;
          best_wg = dot_wgsize;
          best_groups = dot_num_groups;
          best_unroll = dot_unroll;
        }
      }
    }
  }

  use_manual_dot_reduction = best_manual;
  dot_wgsize = best_wg;
  dot_num_groups = best_groups;
  dot_unroll = best_unroll;
}

template <class T>
SYCLStream<T>::~SYCLStream()
{
  sycl::free(d_a, *queue);
  sycl::free(d_b, *queue);
  sycl::free(d_c, *queue);
  sycl::free(d_sum, *queue);
  sycl::free(d_dot, *queue);
  delete[] h_sum;
  devices.clear();
}

template <class T>
void SYCLStream<T>::copy()
{
  const size_t N = array_size;
  T *a = d_a;
  T *c = d_c;
  const size_t wg = copy_wgsize;
  const size_t global_size = round_up(array_size, wg);
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<copy_kernel>(nd_range<1>(global_size, wg), [=](nd_item<1> item)
    {
      const size_t idx = item.get_global_id(0);
      if (idx < N)
      {
        c[idx] = a[idx];
      }
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::mul()
{
  const T scalar = startScalar;
  const size_t N = array_size;
  T *b = d_b;
  T *c = d_c;
  const size_t wg = mul_wgsize;
  const size_t global_size = round_up(array_size, wg);
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<mul_kernel>(nd_range<1>(global_size, wg), [=](nd_item<1> item)
    {
      const size_t idx = item.get_global_id(0);
      if (idx < N)
      {
        b[idx] = scalar * c[idx];
      }
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::add()
{
  const size_t N = array_size;
  T *a = d_a;
  T *b = d_b;
  T *c = d_c;
  const size_t wg = add_wgsize;
  const size_t global_size = round_up(array_size, wg);
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<add_kernel>(nd_range<1>(global_size, wg), [=](nd_item<1> item)
    {
      const size_t idx = item.get_global_id(0);
      if (idx < N)
      {
        c[idx] = a[idx] + b[idx];
      }
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::triad()
{
  const T scalar = startScalar;
  const size_t N = array_size;
  T *a = d_a;
  T *b = d_b;
  T *c = d_c;
  const size_t wg = triad_wgsize;
  const size_t global_size = round_up(array_size, wg);
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<triad_kernel>(nd_range<1>(global_size, wg), [=](nd_item<1> item)
    {
      const size_t idx = item.get_global_id(0);
      if (idx < N)
      {
        a[idx] = b[idx] + scalar * c[idx];
      }
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::nstream()
{
  const T scalar = startScalar;
  const size_t N = array_size;
  T *a = d_a;
  T *b = d_b;
  T *c = d_c;
  const size_t wg = nstream_wgsize;
  const size_t global_size = round_up(array_size, wg);
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<nstream_kernel>(nd_range<1>(global_size, wg), [=](nd_item<1> item)
    {
      const size_t idx = item.get_global_id(0);
      if (idx < N)
      {
        a[idx] += b[idx] + scalar * c[idx];
      }
    });
  });
  queue->wait();
}

template <class T>
T SYCLStream<T>::dot()
{
  if (!use_manual_dot_reduction)
  {
    queue->memset(d_dot, 0, sizeof(T));
    queue->wait();
    const size_t N = array_size;
    T *a = d_a;
    T *b = d_b;
    T *dot = d_dot;

    queue->submit([&](handler &cgh)
    {
      cgh.parallel_for(range<1>{N},
#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
          sycl::reduction(dot, sycl::plus<T>()),
#else
          sycl::reduction(dot, sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
#endif
          [=](id<1> idx, auto& sum)
          {
            sum += a[idx] * b[idx];
          });
    });
    queue->wait();

    T host_dot{};
    queue->memcpy(&host_dot, d_dot, sizeof(T));
    queue->wait();
    return host_dot;
  }

  const size_t N = array_size;
  T *a = d_a;
  T *b = d_b;
  T *sum = d_sum;
  const size_t unroll = dot_unroll;
  queue->submit([&](handler &cgh)
  {
    auto wg_sum = local_accessor<T, 1>(range<1>(dot_wgsize), cgh);

    cgh.parallel_for<dot_kernel>(nd_range<1>(dot_num_groups*dot_wgsize, dot_wgsize), [=](nd_item<1> item)
    {
      size_t i = item.get_global_id(0);
      size_t li = item.get_local_id(0);
      size_t global_size = item.get_global_range()[0];
      const size_t step = global_size * unroll;

      wg_sum[li] = {};
      for (; i + (unroll - 1) * global_size < N; i += step)
      {
        T thread_sum = wg_sum[li];
        if (unroll >= 1) thread_sum += a[i] * b[i];
        if (unroll >= 2) thread_sum += a[i + global_size] * b[i + global_size];
        if (unroll >= 3) thread_sum += a[i + 2 * global_size] * b[i + 2 * global_size];
        if (unroll >= 4) thread_sum += a[i + 3 * global_size] * b[i + 3 * global_size];
        if (unroll >= 5) thread_sum += a[i + 4 * global_size] * b[i + 4 * global_size];
        if (unroll >= 6) thread_sum += a[i + 5 * global_size] * b[i + 5 * global_size];
        if (unroll >= 7) thread_sum += a[i + 6 * global_size] * b[i + 6 * global_size];
        if (unroll >= 8) thread_sum += a[i + 7 * global_size] * b[i + 7 * global_size];
        wg_sum[li] = thread_sum;
      }

      for (; i < N; i += global_size)
      {
        wg_sum[li] += a[i] * b[i];
      }

      size_t local_size = item.get_local_range()[0];
      for (size_t offset = local_size / 2; offset > 0; offset /= 2)
      {
        item.barrier(access::fence_space::local_space);
        if (li < offset)
          wg_sum[li] += wg_sum[li + offset];
      }

      if (li == 0)
        sum[item.get_group(0)] = wg_sum[0];
    });
  });
  queue->wait();

  T final_sum{};
  queue->memcpy(h_sum, d_sum, dot_num_groups * sizeof(T));
  queue->wait();
  for (size_t i = 0; i < dot_num_groups; i++)
  {
    final_sum += h_sum[i];
  }

  return final_sum;
}

template <class T>
void SYCLStream<T>::init_arrays(T initA, T initB, T initC)
{
  const size_t N = array_size;
  T *a = d_a;
  T *b = d_b;
  T *c = d_c;
  const size_t global_size = round_up(array_size, stream_wgsize);
  queue->submit([&](handler &cgh)
  {
    cgh.parallel_for<init_kernel>(nd_range<1>(global_size, stream_wgsize), [=](nd_item<1> item)
    {
      const size_t idx = item.get_global_id(0);
      if (idx < N)
      {
        a[idx] = initA;
        b[idx] = initB;
        c[idx] = initC;
      }
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  queue->memcpy(a.data(), d_a, array_size * sizeof(T));
  queue->memcpy(b.data(), d_b, array_size * sizeof(T));
  queue->memcpy(c.data(), d_c, array_size * sizeof(T));
  queue->wait();
}

void getDeviceList(void)
{
  // Ask SYCL runtime for all devices in system
  devices = sycl::device::get_devices();
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;

  if (device < devices.size())
  {
    name = devices[device].get_info<info::device::name>();
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;
}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    driver = devices[device].get_info<info::device::driver_version>();
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}

// TODO: Fix kernel names to allow multiple template specializations
template class SYCLStream<float>;
template class SYCLStream<double>;
