// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
// Copyright (c) 2022 Troels Henriksen
// University of Copenhagen
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include <string>
#include "FutharkStream.h"

template <class T>
FutharkStream<T>::FutharkStream(const int ARRAY_SIZE, int device)
{
  this->array_size = ARRAY_SIZE;
  this->cfg = futhark_context_config_new();
  this->device = "#" + std::to_string(device);
#if defined(FUTHARK_BACKEND_cuda) || defined(FUTHARK_BACKEND_opencl)
  futhark_context_config_set_device(cfg, this->device.c_str());
#endif
  this->ctx = futhark_context_new(cfg);
  this->a = NULL;
  this->b = NULL;
  this->c = NULL;
}

template <>
FutharkStream<float>::~FutharkStream()
{
  if (this->a) {
    futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->a);
  }
  if (this->b) {
    futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->b);
  }
  if (this->c) {
    futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->c);
  }
  futhark_context_free(this->ctx);
  futhark_context_config_free(this->cfg);
}

template <>
FutharkStream<double>::~FutharkStream()
{
  if (this->a) {
    futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->a);
  }
  if (this->b) {
    futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->b);
  }
  if (this->c) {
    futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->c);
  }
  futhark_context_free(this->ctx);
  futhark_context_config_free(this->cfg);
}

template <>
void FutharkStream<float>::init_arrays(float initA, float initB, float initC) {
  int array_size = this->array_size;
  float *a = new float[array_size];
  float *b = new float[array_size];
  float *c = new float[array_size];
  for (int i = 0; i < array_size; i++) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
  this->a = (futhark_f32_1d*)futhark_new_f32_1d(this->ctx, a, array_size);
  this->b = (futhark_f32_1d*)futhark_new_f32_1d(this->ctx, b, array_size);
  this->c = (futhark_f32_1d*)futhark_new_f32_1d(this->ctx, c, array_size);
  futhark_context_sync(this->ctx);
  delete[] a;
  delete[] b;
  delete[] c;
}

template <>
void FutharkStream<double>::init_arrays(double initA, double initB, double initC) {
  int array_size = this->array_size;
  double *a = new double[array_size];
  double *b = new double[array_size];
  double *c = new double[array_size];
  for (int i = 0; i < array_size; i++) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
  this->a = (futhark_f64_1d*)futhark_new_f64_1d(this->ctx, a, array_size);
  this->b = (futhark_f64_1d*)futhark_new_f64_1d(this->ctx, b, array_size);
  this->c = (futhark_f64_1d*)futhark_new_f64_1d(this->ctx, c, array_size);
  futhark_context_sync(this->ctx);
  delete[] a;
  delete[] b;
  delete[] c;
}

template <>
void FutharkStream<float>::read_arrays(std::vector<float>& h_a, std::vector<float>& h_b, std::vector<float>& h_c) {
  futhark_values_f32_1d(this->ctx, (futhark_f32_1d*)this->a, h_a.data());
  futhark_values_f32_1d(this->ctx, (futhark_f32_1d*)this->b, h_b.data());
  futhark_values_f32_1d(this->ctx, (futhark_f32_1d*)this->c, h_c.data());
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<double>::read_arrays(std::vector<double>& h_a, std::vector<double>& h_b, std::vector<double>& h_c) {
  futhark_values_f64_1d(this->ctx, (futhark_f64_1d*)this->a, h_a.data());
  futhark_values_f64_1d(this->ctx, (futhark_f64_1d*)this->b, h_b.data());
  futhark_values_f64_1d(this->ctx, (futhark_f64_1d*)this->c, h_c.data());
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<float>::copy() {
  futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->c);
  futhark_entry_f32_copy(this->ctx, (futhark_f32_1d**)&this->c, (futhark_f32_1d*)this->a);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<double>::copy() {
  futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->c);
  futhark_entry_f64_copy(this->ctx, (futhark_f64_1d**)&this->c, (futhark_f64_1d*)this->a);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<float>::mul() {
  futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->b);
  futhark_entry_f32_mul(this->ctx, (futhark_f32_1d**)&this->b, (futhark_f32_1d*)this->c);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<double>::mul() {
  futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->b);
  futhark_entry_f64_mul(this->ctx, (futhark_f64_1d**)&this->b, (futhark_f64_1d*)this->c);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<float>::add() {
  futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->c);
  futhark_entry_f32_add(this->ctx, (futhark_f32_1d**)&this->c, (futhark_f32_1d*)this->a, (futhark_f32_1d*)this->b);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<double>::add() {
  futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->c);
  futhark_entry_f64_add(this->ctx, (futhark_f64_1d**)&this->c, (futhark_f64_1d*)this->a, (futhark_f64_1d*)this->b);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<float>::triad() {
  futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->c);
  futhark_entry_f32_triad(this->ctx, (futhark_f32_1d**)&this->c, (futhark_f32_1d*)this->a, (futhark_f32_1d*)this->b);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<double>::triad() {
  futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->a);
  futhark_entry_f64_triad(this->ctx, (futhark_f64_1d**)&this->a, (futhark_f64_1d*)this->b, (futhark_f64_1d*)this->c);
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<float>::nstream() {
  futhark_f32_1d* d;
  futhark_entry_f32_triad(this->ctx, &d, (futhark_f32_1d*)this->a, (futhark_f32_1d*)this->b);
  futhark_free_f32_1d(this->ctx, (futhark_f32_1d*)this->c);
  this->c = d;
  futhark_context_sync(this->ctx);
}

template <>
void FutharkStream<double>::nstream() {
  futhark_f64_1d* d;
  futhark_entry_f64_triad(this->ctx, &d, (futhark_f64_1d*)this->a, (futhark_f64_1d*)this->b);
  futhark_free_f64_1d(this->ctx, (futhark_f64_1d*)this->c);
  this->c = d;
  futhark_context_sync(this->ctx);
}

template <>
float FutharkStream<float>::dot() {
  float res;
  futhark_entry_f32_dot(this->ctx, &res, (futhark_f32_1d*)this->a, (futhark_f32_1d*)this->b);
  futhark_context_sync(this->ctx);
  return res;
}

template <>
double FutharkStream<double>::dot() {
  double res;
  futhark_entry_f64_dot(this->ctx, &res, (futhark_f64_1d*)this->a, (futhark_f64_1d*)this->b);
  futhark_context_sync(this->ctx);
  return res;
}

void listDevices(void)
{
  std::cout << "Device selection not supported." << std::endl;
}

template class FutharkStream<float>;
template class FutharkStream<double>;
