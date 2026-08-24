#include "HostKernels.h"

#include "jit/Jit.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tvarant::host {
namespace {

constexpr int64_t kTile = 32;

}  // namespace

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

void silu_f32(const float* in, float* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const float x = in[i];
    out[i] = x / (1.f + std::exp(-x));
  }
}

void scale_f32(const float* in, float* out, int64_t n, float scale) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = in[i] * scale;
  }
}

void add_scalar_f32(const float* in, float* out, int64_t n, float scalar) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = in[i] + scalar;
  }
}

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
    bool trans_b) {
  for (int64_t i0 = 0; i0 < m; i0 += kTile) {
    const int64_t i1 = std::min(i0 + kTile, m);
    for (int64_t j0 = 0; j0 < n; j0 += kTile) {
      const int64_t j1 = std::min(j0 + kTile, n);
      float acc[kTile][kTile];
      std::memset(acc, 0, sizeof(acc));
      for (int64_t p0 = 0; p0 < k; p0 += kTile) {
        const int64_t p1 = std::min(p0 + kTile, k);
        for (int64_t i = i0; i < i1; ++i) {
          for (int64_t p = p0; p < p1; ++p) {
            const float av = a[i * k + p];
            for (int64_t j = j0; j < j1; ++j) {
              const float bv = trans_b ? b[j * k + p] : b[p * n + j];
              acc[i - i0][j - j0] += av * bv;
            }
          }
        }
      }
      for (int64_t i = i0; i < i1; ++i) {
        for (int64_t j = j0; j < j1; ++j) {
          float prev = 0.f;
          if (bias != nullptr) {
            prev = bias[j];
          } else if (beta != 0.f) {
            prev = c[i * n + j];
          }
          float v = alpha * acc[i - i0][j - j0] + beta * prev;
          c[i * n + j] = jit::apply_act(v, act);
        }
      }
    }
  }
}

void bmm_f32(
    const float* a,
    const float* b,
    float* c,
    int64_t batch,
    int64_t m,
    int64_t n,
    int64_t k,
    bool trans_b) {
  const int64_t a_stride = m * k;
  const int64_t b_stride = trans_b ? n * k : k * n;
  const int64_t c_stride = m * n;
  for (int64_t bi = 0; bi < batch; ++bi) {
    gemm_bias_act_f32(
        a + bi * a_stride,
        b + bi * b_stride,
        nullptr,
        c + bi * c_stride,
        m,
        n,
        k,
        1.f,
        0.f,
        0,
        trans_b);
  }
}

void softmax_f32(
    const float* in,
    float* out,
    int64_t outer,
    int64_t dim,
    int64_t inner) {
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      float m = in[(o * dim) * inner + i];
      for (int64_t d = 1; d < dim; ++d) {
        m = std::max(m, in[(o * dim + d) * inner + i]);
      }
      float sum = 0.f;
      for (int64_t d = 0; d < dim; ++d) {
        const float e = std::exp(in[(o * dim + d) * inner + i] - m);
        out[(o * dim + d) * inner + i] = e;
        sum += e;
      }
      const float inv = 1.f / sum;
      for (int64_t d = 0; d < dim; ++d) {
        out[(o * dim + d) * inner + i] *= inv;
      }
    }
  }
}

void layer_norm_f32(
    const float* in,
    const float* weight,
    const float* bias,
    float* out,
    float* mean,
    float* rstd,
    int64_t outer,
    int64_t n,
    float eps) {
  for (int64_t r = 0; r < outer; ++r) {
    const float* row = in + r * n;
    float mu = 0.f;
    for (int64_t j = 0; j < n; ++j) {
      mu += row[j];
    }
    mu /= static_cast<float>(n);
    float var = 0.f;
    for (int64_t j = 0; j < n; ++j) {
      const float d = row[j] - mu;
      var += d * d;
    }
    var /= static_cast<float>(n);
    const float rs = 1.f / std::sqrt(var + eps);
    if (mean != nullptr) {
      mean[r] = mu;
    }
    if (rstd != nullptr) {
      rstd[r] = rs;
    }
    float* orow = out + r * n;
    for (int64_t j = 0; j < n; ++j) {
      float y = (row[j] - mu) * rs;
      if (weight != nullptr) {
        y *= weight[j];
      }
      if (bias != nullptr) {
        y += bias[j];
      }
      orow[j] = y;
    }
  }
}

void embedding_f32(
    const float* weight,
    const int32_t* indices,
    float* out,
    int64_t n_idx,
    int64_t dim,
    int64_t vocab) {
  for (int64_t i = 0; i < n_idx; ++i) {
    const int32_t id = indices[i];
    float* dst = out + i * dim;
    if (id < 0 || static_cast<int64_t>(id) >= vocab) {
      std::memset(dst, 0, static_cast<size_t>(dim) * sizeof(float));
      continue;
    }
    std::memcpy(dst, weight + static_cast<int64_t>(id) * dim, static_cast<size_t>(dim) * sizeof(float));
  }
}

}  // namespace tvarant::host
