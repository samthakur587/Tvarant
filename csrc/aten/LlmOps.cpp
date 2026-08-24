#include "aten/AtenCommon.h"
#include "runtime/TvarantRuntime.h"

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tvarant::ops {

at::Tensor silu(const at::Tensor& self) {
  check_fp32(self, "silu");
  auto in = self.contiguous();
  auto out = at::empty(in.sizes(), in.options());
  LaunchParams p;
  p.kernel = "silu_kernel";
  p.src0 = in.data_ptr();
  p.dst = out.data_ptr();
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor& silu_(at::Tensor& self) {
  auto out = silu(self);
  self.copy_(out);
  return self;
}

at::Tensor& silu_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = silu(self);
  out.copy_(tmp);
  return out;
}

at::Tensor mul_scalar(const at::Tensor& self, const at::Scalar& other) {
  check_fp32(self, "mul");
  auto in = self.contiguous();
  auto out = at::empty(in.sizes(), in.options());
  LaunchParams p;
  p.kernel = "scale_kernel";
  p.src0 = in.data_ptr();
  p.dst = out.data_ptr();
  p.scalar = other.toFloat();
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor add_scalar(
    const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  check_fp32(self, "add");
  auto in = self.contiguous();
  auto out = at::empty(in.sizes(), in.options());
  LaunchParams p;
  p.kernel = "add_scalar_kernel";
  p.src0 = in.data_ptr();
  p.dst = out.data_ptr();
  p.scalar = other.toFloat() * alpha.toFloat();
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor _softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  (void)half_to_float;
  check_fp32(self, "softmax");
  TORCH_CHECK(self.dim() > 0, "softmax: expected non-scalar");
  if (dim < 0) {
    dim += self.dim();
  }
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "softmax dim out of range");
  auto in = self.contiguous();
  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer *= in.size(i);
  }
  const int64_t d = in.size(dim);
  for (int64_t i = dim + 1; i < in.dim(); ++i) {
    inner *= in.size(i);
  }
  auto out = at::empty(in.sizes(), in.options());
  LaunchParams p;
  p.kernel = "softmax_kernel";
  p.src0 = in.data_ptr();
  p.dst = out.data_ptr();
  p.m = outer;
  p.n = d;
  p.k = inner;
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor softmax_int(
    const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype) {
  TORCH_CHECK(
      !dtype.has_value() || dtype.value() == at::kFloat,
      "Tvarant softmax supports float32 only");
  return _softmax(self, dim, false);
}

at::Tensor matmul(const at::Tensor& self, const at::Tensor& other) {
  check_fp32(self, "matmul");
  check_fp32(other, "matmul");
  TORCH_CHECK(self.dim() >= 2 && other.dim() >= 2, "matmul: expected at least 2D tensors");
  if (self.dim() == 2 && other.dim() == 2) {
    return gemm_bias_act(self, other, nullptr, 1.f, 0.f, 0, false);
  }
  auto a = self.contiguous();
  auto b = other.contiguous();
  TORCH_CHECK(a.size(-1) == b.size(-2), "matmul: contracting dim mismatch ", a.sizes(), " vs ", b.sizes());
  const int64_t k = a.size(-1);
  const int64_t m = a.size(-2);
  const int64_t n = b.size(-1);
  const int64_t a_batch = a.numel() / (m * k);
  int64_t b_batch = b.numel() / (k * n);
  at::Tensor b_use = b;
  if (b.dim() == 2 && a_batch > 1) {
    b_use = b.unsqueeze(0).expand({a_batch, b.size(0), b.size(1)}).contiguous();
    b_batch = a_batch;
  }
  TORCH_CHECK(a_batch == b_batch, "matmul: batch mismatch ", a.sizes(), " vs ", b.sizes());
  auto out_sizes = a.sizes().vec();
  out_sizes.back() = n;
  auto out = at::empty(out_sizes, a.options());
  LaunchParams p;
  p.kernel = "bmm_kernel";
  p.src0 = a.data_ptr();
  p.src1 = b_use.data_ptr();
  p.dst = out.data_ptr();
  p.batch = a_batch;
  p.m = m;
  p.n = n;
  p.k = k;
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor bmm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "bmm(): tensors must be 3D");
  return matmul(self, mat2);
}

at::Tensor linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt) {
  check_fp32(input, "linear");
  check_fp32(weight, "linear");
  TORCH_CHECK(weight.dim() == 2, "linear: weight must be 2D");
  auto x = input.contiguous();
  const int64_t in_f = x.size(-1);
  TORCH_CHECK(in_f == weight.size(1), "linear: in_features mismatch");
  const int64_t out_f = weight.size(0);
  auto out_sizes = x.sizes().vec();
  out_sizes.back() = out_f;
  const int64_t m = x.numel() / in_f;
  auto x2 = x.reshape({m, in_f});
  const at::Tensor* bias = nullptr;
  at::Tensor bias_store;
  if (bias_opt.has_value() && bias_opt->defined()) {
    check_fp32(*bias_opt, "linear");
    bias_store = *bias_opt;
    bias = &bias_store;
  }
  auto y2 = gemm_bias_act(x2, weight, bias, 1.f, 1.f, 0, true);
  return y2.reshape(out_sizes);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double eps) {
  check_fp32(input, "layer_norm");
  int64_t n = 1;
  for (const auto d : normalized_shape) {
    n *= d;
  }
  TORCH_CHECK(n > 0 && input.numel() % n == 0, "layer_norm: invalid normalized_shape");
  const int64_t outer = input.numel() / n;
  auto in = input.contiguous();
  auto out = at::empty(in.sizes(), in.options());
  auto mean = at::empty({outer}, in.options());
  auto rstd = at::empty({outer}, in.options());
  at::Tensor w_c;
  at::Tensor b_c;
  const void* wptr = nullptr;
  const void* bptr = nullptr;
  if (weight.has_value() && weight->defined()) {
    check_fp32(*weight, "layer_norm");
    w_c = weight->contiguous();
    wptr = w_c.data_ptr();
  }
  if (bias.has_value() && bias->defined()) {
    check_fp32(*bias, "layer_norm");
    b_c = bias->contiguous();
    bptr = b_c.data_ptr();
  }
  LaunchParams p;
  p.kernel = "layernorm_kernel";
  p.src0 = in.data_ptr();
  p.src1 = wptr;
  p.src2 = bptr;
  p.dst = out.data_ptr();
  p.aux0 = mean.data_ptr();
  p.aux1 = rstd.data_ptr();
  p.m = outer;
  p.n = n;
  p.scalar = static_cast<float>(eps);
  ::tvarant::runtime().launch(p);
  auto mean_sizes = in.sizes().vec();
  TORCH_CHECK(
      mean_sizes.size() >= normalized_shape.size(),
      "layer_norm: normalized_shape longer than input");
  mean_sizes.resize(mean_sizes.size() - normalized_shape.size());
  return {out, mean.reshape(mean_sizes), rstd.reshape(mean_sizes)};
}

at::Tensor embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  (void)padding_idx;
  (void)scale_grad_by_freq;
  (void)sparse;
  check_fp32(weight, "embedding");
  TORCH_CHECK(weight.dim() == 2, "embedding: weight must be 2D");
  auto idx_cpu = indices.to(at::kCPU).to(at::kInt).contiguous();
  auto idx_dev = at::empty(
      idx_cpu.sizes(), weight.options().dtype(at::kInt));
  if (idx_cpu.numel() > 0) {
    ::tvarant::runtime().copy_h2d(
        idx_dev.data_ptr(),
        idx_cpu.data_ptr(),
        static_cast<size_t>(idx_cpu.numel()) * sizeof(int32_t));
  }
  auto out_sizes = indices.sizes().vec();
  out_sizes.push_back(weight.size(1));
  auto w = weight.contiguous();
  auto out = at::empty(out_sizes, w.options());
  if (idx_dev.numel() == 0 || w.size(1) == 0) {
    return out;
  }
  LaunchParams p;
  p.kernel = "embedding_kernel";
  p.src0 = w.data_ptr();
  p.src1 = idx_dev.data_ptr();
  p.dst = out.data_ptr();
  p.numel = idx_dev.numel();
  p.n = w.size(1);
  p.k = w.size(0);
  ::tvarant::runtime().launch(p);
  return out;
}

}  // namespace tvarant::ops

namespace {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> wrap_native_layer_norm(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double eps) {
  return tvarant::ops::native_layer_norm(
      input, C10_AS_INTARRAYREF_SLOW(normalized_shape), weight, bias, eps);
}

int g_tvarant_llm_force_link = 0;

}  // namespace

int tvarant_llm_force_link() {
  return g_tvarant_llm_force_link;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("silu", TORCH_FN(tvarant::ops::silu));
  m.impl("silu_", TORCH_FN(tvarant::ops::silu_));
  m.impl("silu.out", TORCH_FN(tvarant::ops::silu_out));
  m.impl("mul.Scalar", TORCH_FN(tvarant::ops::mul_scalar));
  m.impl("add.Scalar", TORCH_FN(tvarant::ops::add_scalar));
  m.impl("_softmax", TORCH_FN(tvarant::ops::_softmax));
  m.impl("softmax.int", TORCH_FN(tvarant::ops::softmax_int));
  m.impl("matmul", TORCH_FN(tvarant::ops::matmul));
  m.impl("bmm", TORCH_FN(tvarant::ops::bmm));
  m.impl("linear", TORCH_FN(tvarant::ops::linear));
  m.impl("native_layer_norm", TORCH_FN(wrap_native_layer_norm));
  m.impl("embedding", TORCH_FN(tvarant::ops::embedding));
}
