#include "HostKernels.h"

#include <algorithm>
#include <cstring>

namespace tvarant::host {

void fill_f32(float* out, float value, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = value;
  }
}

void copy_f32(float* dst, const float* src, int64_t n) {
  std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

void add_f32(const float* a, const float* b, float* out, int64_t n, float alpha) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = a[i] + alpha * b[i];
  }
}

void mul_f32(const float* a, const float* b, float* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
}

void relu_f32(const float* in, float* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = in[i] > 0.f ? in[i] : 0.f;
  }
}

void gemm_f32(
    const float* a,
    const float* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    float beta) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.f;
      for (int64_t p = 0; p < k; ++p) {
        acc += a[i * k + p] * b[p * n + j];
      }
      const float prev = (beta == 0.f) ? 0.f : c[i * n + j];
      c[i * n + j] = alpha * acc + beta * prev;
    }
  }
}

}  // namespace tvarant::host
