#pragma once

#include <cstdint>

namespace tvarant::host {

void fill_f32(float* out, float value, int64_t n);
void arange_f32(float* out, int64_t n, float start, float step);
void copy_f32(float* dst, const float* src, int64_t n);
void add_f32(const float* a, const float* b, float* out, int64_t n, float alpha);
void mul_f32(const float* a, const float* b, float* out, int64_t n);
void relu_f32(const float* in, float* out, int64_t n);
void silu_f32(const float* in, float* out, int64_t n);
void scale_f32(const float* in, float* out, int64_t n, float scale);
void add_scalar_f32(const float* in, float* out, int64_t n, float scalar);
void neg_f32(const float* in, float* out, int64_t n);
void abs_f32(const float* in, float* out, int64_t n);
void exp_f32(const float* in, float* out, int64_t n);
void exp2_f32(const float* in, float* out, int64_t n);
void sqrt_f32(const float* in, float* out, int64_t n);
void rsqrt_f32(const float* in, float* out, int64_t n);

void gemm_bias_act_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    float beta,
    int act,
    bool trans_b);

inline void gemm_f32(
    const float* a,
    const float* b,
    float* c,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    float beta) {
  gemm_bias_act_f32(a, b, nullptr, c, m, n, k, alpha, beta, 0, false);
}

void bmm_f32(
    const float* a,
    const float* b,
    float* c,
    int64_t batch,
    int64_t m,
    int64_t n,
    int64_t k,
    bool trans_b);

void softmax_f32(
    const float* in,
    float* out,
    int64_t outer,
    int64_t dim,
    int64_t inner);

void layer_norm_f32(
    const float* in,
    const float* weight,
    const float* bias,
    float* out,
    float* mean,
    float* rstd,
    int64_t outer,
    int64_t n,
    float eps);

void embedding_f32(
    const float* weight,
    const int32_t* indices,
    float* out,
    int64_t n_idx,
    int64_t dim,
    int64_t vocab);

}  // namespace tvarant::host
