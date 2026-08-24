#pragma once

#include "runtime/TvarantRuntime.h"

#include <ATen/ATen.h>

namespace tvarant::ops {

inline void check_fp32(const at::Tensor& t, const char* op) {
  TORCH_CHECK(
      t.scalar_type() == at::kFloat,
      "Tvarant op '",
      op,
      "' supports float32 only, got ",
      t.scalar_type());
}

// Float32-only device: quietly downcast float64 (common when Python floats
// promote) so mul/add with bare scalars keep working.
inline at::Tensor ensure_fp32(const at::Tensor& t, const char* op) {
  if (t.scalar_type() == at::kFloat) {
    return t;
  }
  if (t.scalar_type() == at::kDouble) {
    return t.to(at::kFloat);
  }
  check_fp32(t, op);
  return t;
}

// C[m,n] = act(alpha * A[m,k] @ B + beta * bias[n])
// B is [k,n] if trans_b is false, else [n,k].
inline at::Tensor gemm_bias_act(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor* bias,
    float alpha,
    float beta,
    int act,
    bool trans_b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm: expected 2D tensors");
  auto ac = a.contiguous();
  auto bc = b.contiguous();
  const int64_t m = ac.size(0);
  const int64_t k = ac.size(1);
  const int64_t n = trans_b ? bc.size(0) : bc.size(1);
  if (trans_b) {
    TORCH_CHECK(bc.size(1) == k, "gemm trans_b: K mismatch ", bc.sizes(), " vs K=", k);
  } else {
    TORCH_CHECK(bc.size(0) == k, "gemm: K mismatch ", ac.sizes(), " vs ", bc.sizes());
  }
  at::Tensor bias_c;
  const void* bias_ptr = nullptr;
  float beta_use = 0.f;
  if (bias != nullptr && beta != 0.f) {
    bias_c = bias->contiguous();
    TORCH_CHECK(
        bias_c.numel() == n,
        "gemm bias must have ",
        n,
        " elements, got ",
        bias_c.numel());
    bias_ptr = bias_c.data_ptr();
    beta_use = beta;
  }
  auto out = at::empty({m, n}, ac.options());
  if (m == 0 || n == 0) {
    return out;
  }
  LaunchParams p;
  p.kernel = "gemm_bias_act_kernel";
  p.src0 = ac.data_ptr();
  p.src1 = bc.data_ptr();
  p.src2 = bias_ptr;
  p.dst = out.data_ptr();
  p.m = m;
  p.n = n;
  p.k = k;
  p.alpha = alpha;
  p.beta = beta_use;
  p.act = act;
  p.trans_b = trans_b;
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

}  // namespace tvarant::ops
