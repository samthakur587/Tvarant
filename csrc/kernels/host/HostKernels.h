#pragma once

#include <cstdint>

namespace tvarant::host {

void fill_f32(float* out, float value, int64_t n);
void copy_f32(float* dst, const float* src, int64_t n);
void add_f32(const float* a, const float* b, float* out, int64_t n, float alpha);
void mul_f32(const float* a, const float* b, float* out, int64_t n);
void relu_f32(const float* in, float* out, int64_t n);
void gemm_f32(
    const float* a,
    const float* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    float beta);

}  // namespace tvarant::host
