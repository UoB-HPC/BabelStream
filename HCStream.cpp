// Copyright (c) 2017 Peter Steinbach, MPI CBG Scientific Computing Facility
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "HCStream.h"

#include <codecvt>
#include <vector>
#include <locale>
#include <numeric>

//specific sizes were obtained through experimentation using a Fiji R9 Nano with rocm 1.6-115
#ifndef VIRTUALTILESIZE
#define VIRTUALTILESIZE 256
#endif

//specific sizes were obtained through experimentation using a Fiji R9 Nano with rocm 1.6-115
#ifndef NTILES
#define NTILES 2048
#endif


std::string getDeviceName(const hc::accelerator& _acc)
{
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  std::string value = converter.to_bytes(_acc.get_description());
  return value;
}

void listDevices(void)
{
  // Get number of devices
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();

  // Print device names
  if (accs.empty())
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < accs.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(accs[i]) << std::endl;
    }
    std::cout << std::endl;
  }
}


template <class T>
HCStream<T>::HCStream(const int ARRAY_SIZE, const int device_index):
  array_size(ARRAY_SIZE),
  d_a(ARRAY_SIZE),
  d_b(ARRAY_SIZE),
  d_c(ARRAY_SIZE)
{

  // The array size must be divisible by VIRTUALTILESIZE for kernel launches
  if (ARRAY_SIZE % VIRTUALTILESIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << VIRTUALTILESIZE;
    throw std::runtime_error(ss.str());
  }

  // Set device
  std::vector<hc::accelerator> accs = hc::accelerator::get_all();
  auto current = accs.at(device_index);

  hc::accelerator::set_default(current.get_device_path());

  std::cout << "Using HC device " << getDeviceName(current) << std::endl;

}


template <class T>
HCStream<T>::~HCStream()
{
}

template <class T>
void HCStream<T>::init_arrays(T _a, T _b, T _c)
{
  hc::array_view<T,1> view_a(this->d_a);
  hc::array_view<T,1> view_b(this->d_b);
  hc::array_view<T,1> view_c(this->d_c);

  hc::completion_future future_a= hc::parallel_for_each(hc::extent<1>(array_size)
                                , [=](hc::index<1> i) [[hc]] {
                                  view_a[i] = _a;
                                });

  hc::completion_future future_b= hc::parallel_for_each(hc::extent<1>(array_size)
                                                        , [=](hc::index<1> i) [[hc]] {
                                                          view_b[i] = _b;
                                                        });
  hc::completion_future future_c= hc::parallel_for_each(hc::extent<1>(array_size)
                                                        , [=](hc::index<1> i) [[hc]] {
                                                          view_c[i] = _c;
                                                        });
  try{
    future_a.wait();
    future_b.wait();
    future_c.wait();
  }
  catch(std::exception& e){
    std::cerr << __FILE__ << ":" << __LINE__ << "\t HCStream<T>::init_arrays " << e.what() << std::endl;
    throw;
  }

}

template <class T>
void HCStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  hc::copy(d_a,a.begin());
  hc::copy(d_b,b.begin());
  hc::copy(d_c,c.begin());
}


template <class T>
void HCStream<T>::copy()
{

  hc::array_view<T,1> view_a = this->d_a;
  hc::array_view<T,1> view_c = this->d_c;

  try{
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
                                , [=](hc::index<1> index) [[hc]] {
                                  view_c[index] = view_a[index];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cerr << __FILE__ << ":" << __LINE__ << "\t HCStream<T>::copy " << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::mul()
{

  const T scalar = startScalar;
  hc::array_view<T,1> view_b = this->d_b;
  hc::array_view<T,1> view_c = this->d_c;

  try{
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
                                , [=](hc::index<1> i) [[hc]] {
                                  view_b[i] = scalar*view_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cerr << __FILE__ << ":" << __LINE__ << "\t HCStream<T>::mul " << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::add()
{


  hc::array_view<T,1> view_a(this->d_a);
  hc::array_view<T,1> view_b(this->d_b);
  hc::array_view<T,1> view_c(this->d_c);

  try{
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
                                , [=](hc::index<1> i) [[hc]] {
                                  view_c[i] = view_a[i]+view_b[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cerr << __FILE__ << ":" << __LINE__ << "\t HCStream<T>::add " << e.what() << std::endl;
    throw;
  }
}

template <class T>
void HCStream<T>::triad()
{

  const T scalar = startScalar;
  hc::array_view<T,1> view_a(this->d_a);
  hc::array_view<T,1> view_b(this->d_b);
  hc::array_view<T,1> view_c(this->d_c);

  try{
    hc::completion_future future_kernel = hc::parallel_for_each(hc::extent<1>(array_size)
                                , [=](hc::index<1> i) [[hc]] {
                                  view_a[i] = view_b[i] + scalar*view_c[i];
								});
    future_kernel.wait();
  }
  catch(std::exception& e){
    std::cerr << __FILE__ << ":" << __LINE__ << "\t HCStream<T>::triad " << e.what() << std::endl;
    throw;
  }
}

template <class T>
T HCStream<T>::dot()
{
   //implementation adapted from
    //https://ampbook.codeplex.com/SourceControl/latest
    // ->Samples/CaseStudies/Reduction
    // ->CascadingReduction.h

    const auto& view_a = this->d_a;
    const auto& view_b = this->d_b;

    auto ex = view_a.get_extent();
    const auto tiled_ex = hc::extent<1>(NTILES * VIRTUALTILESIZE).tile(VIRTUALTILESIZE);
    const auto domain_sz = tiled_ex.size();

    hc::array<T, 1> partial(NTILES);

    hc::parallel_for_each(tiled_ex,
                          [=,
                           &view_a,
                           &view_b,
                           &partial](const hc::tiled_index<1>& tidx) [[hc]] {

                            auto gidx = tidx.global[0];
        T r = T{0}; // Assumes reduction op is addition.
        while (gidx < view_a.get_extent().size()) {
            r += view_a[gidx] * view_b[gidx];
            gidx += domain_sz;
        }

        tile_static T tileData[VIRTUALTILESIZE];
        tileData[tidx.local[0]] = r;

        tidx.barrier.wait_with_tile_static_memory_fence();

        for (auto h = VIRTUALTILESIZE / 2; h; h /= 2) {
            if (tidx.local[0] < h) {
                tileData[tidx.local[0]] += tileData[tidx.local[0] + h];
            }
            tidx.barrier.wait_with_tile_static_memory_fence();
        }

        if (tidx.global == tidx.tile_origin) partial[tidx.tile] = tileData[0];
    });

    try {
        partial.get_accelerator_view().wait();
    }
    catch (std::exception& e) {
        std::cerr << __FILE__ << ":" << __LINE__ << "\t  HCStream<T>::dot " << e.what() << std::endl;
        throw;
    }

    std::vector<T> h_partial(NTILES,0);
    hc::copy(partial,h_partial.begin());

    T result = std::accumulate(h_partial.begin(), h_partial.end(), 0.);

    return result;


}

template class HCStream<float>;
template class HCStream<double>;
