#include "aten/AtenCommon.h"
#include "aten/AtenOps.h"
#include "runtime/TvarantRuntime.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>

namespace tvarant::ops {

// ---- factory (#16–#18, #23) ----

at::Tensor zeros(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  auto t = empty_memory_format(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  fill__scalar(t, 0);
  return t;
}

at::Tensor zeros_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto dtype = dtype_opt.value_or(self.scalar_type());
  const auto layout = layout_opt.value_or(self.layout());
  const auto device = device_opt.value_or(self.device());
  auto t = empty_memory_format(
      self.sizes(), dtype, layout, device, pin_memory_opt, memory_format_opt);
  fill__scalar(t, 0);
  return t;
}

at::Tensor ones(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  auto t = empty_memory_format(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  fill__scalar(t, 1);
  return t;
}

at::Tensor ones_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto dtype = dtype_opt.value_or(self.scalar_type());
  const auto layout = layout_opt.value_or(self.layout());
  const auto device = device_opt.value_or(self.device());
  auto t = empty_memory_format(
      self.sizes(), dtype, layout, device, pin_memory_opt, memory_format_opt);
  fill__scalar(t, 1);
  return t;
}

at::Tensor full(
    c10::IntArrayRef size,
    const at::Scalar& fill_value,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  auto t = empty_memory_format(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  fill__scalar(t, fill_value);
  return t;
}

at::Tensor full_like(
    const at::Tensor& self,
    const at::Scalar& fill_value,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto dtype = dtype_opt.value_or(self.scalar_type());
  const auto layout = layout_opt.value_or(self.layout());
  const auto device = device_opt.value_or(self.device());
  auto t = empty_memory_format(
      self.sizes(), dtype, layout, device, pin_memory_opt, memory_format_opt);
  fill__scalar(t, fill_value);
  return t;
}

at::Tensor eye_m(
    int64_t n,
    int64_t m,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  auto out = zeros({n, m}, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  const int64_t diag = std::min(n, m);
  if (diag == 0) {
    return out;
  }
  if (out.scalar_type() == at::kFloat && out.is_contiguous()) {
    LaunchParams p;
    p.kernel = "eye_kernel";
    p.dst = out.data_ptr();
    p.m = n;
    p.n = m;
    p.numel = diag;
    ::tvarant::runtime().launch(p);
    return out;
  }
  auto cpu = at::eye(
      n, m, at::TensorOptions().dtype(out.scalar_type()).device(at::kCPU));
  return ::tvarant::ops::_copy_from(cpu, out, false);
}

at::Tensor eye(
    int64_t n,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  return eye_m(n, n, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ---- clone / contiguous (#19–#20) ----

at::Tensor clone(
    const at::Tensor& self, std::optional<c10::MemoryFormat> memory_format) {
  at::Tensor out;
  if (memory_format.has_value() && *memory_format != c10::MemoryFormat::Preserve) {
    out = empty_memory_format(
        self.sizes(),
        self.scalar_type(),
        self.layout(),
        self.device(),
        false,
        memory_format);
  } else if (self.is_non_overlapping_and_dense()) {
    out = empty_strided(
        self.sizes(),
        self.strides(),
        self.scalar_type(),
        self.layout(),
        self.device(),
        false);
  } else {
    out = empty_memory_format(
        self.sizes(),
        self.scalar_type(),
        self.layout(),
        self.device(),
        false,
        c10::MemoryFormat::Contiguous);
  }
  return ::tvarant::ops::_copy_from(self, out, false);
}

at::Tensor contiguous(const at::Tensor& self, c10::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  auto out = empty_memory_format(
      self.sizes(),
      self.scalar_type(),
      self.layout(),
      self.device(),
      false,
      memory_format);
  return ::tvarant::ops::_copy_from(self, out, false);
}

// ---- unary (#25–#27, #29) ----

namespace {

at::Tensor unary_kernel(const at::Tensor& self, const char* op, const char* kernel) {
  auto in = ensure_fp32(self, op).contiguous();
  auto out = at::empty(in.sizes(), in.options());
  LaunchParams p;
  p.kernel = kernel;
  p.src0 = in.data_ptr();
  p.dst = out.data_ptr();
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

}  // namespace

at::Tensor neg(const at::Tensor& self) {
  return unary_kernel(self, "neg", "neg_kernel");
}

at::Tensor& neg_(at::Tensor& self) {
  auto out = ::tvarant::ops::neg(self);
  ::tvarant::ops::_copy_from(out, self, false);
  return self;
}

at::Tensor& neg_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = ::tvarant::ops::neg(self);
  ::tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor abs(const at::Tensor& self) {
  return unary_kernel(self, "abs", "abs_kernel");
}

at::Tensor& abs_(at::Tensor& self) {
  auto out = ::tvarant::ops::abs(self);
  ::tvarant::ops::_copy_from(out, self, false);
  return self;
}

at::Tensor& abs_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = ::tvarant::ops::abs(self);
  ::tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor exp(const at::Tensor& self) {
  return unary_kernel(self, "exp", "exp_kernel");
}

at::Tensor& exp_(at::Tensor& self) {
  auto out = ::tvarant::ops::exp(self);
  ::tvarant::ops::_copy_from(out, self, false);
  return self;
}

at::Tensor& exp_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = ::tvarant::ops::exp(self);
  ::tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor exp2(const at::Tensor& self) {
  return unary_kernel(self, "exp2", "exp2_kernel");
}

at::Tensor& exp2_(at::Tensor& self) {
  auto out = ::tvarant::ops::exp2(self);
  ::tvarant::ops::_copy_from(out, self, false);
  return self;
}

at::Tensor& exp2_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = ::tvarant::ops::exp2(self);
  ::tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor sqrt(const at::Tensor& self) {
  return unary_kernel(self, "sqrt", "sqrt_kernel");
}

at::Tensor& sqrt_(at::Tensor& self) {
  auto out = ::tvarant::ops::sqrt(self);
  ::tvarant::ops::_copy_from(out, self, false);
  return self;
}

at::Tensor& sqrt_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = ::tvarant::ops::sqrt(self);
  ::tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor rsqrt(const at::Tensor& self) {
  return unary_kernel(self, "rsqrt", "rsqrt_kernel");
}

at::Tensor& rsqrt_(at::Tensor& self) {
  auto out = ::tvarant::ops::rsqrt(self);
  ::tvarant::ops::_copy_from(out, self, false);
  return self;
}

at::Tensor& rsqrt_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = ::tvarant::ops::rsqrt(self);
  ::tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

// ---- sub (#39) ----

at::Tensor sub_tensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return add_tensor(self, other, -alpha.toDouble());
}

at::Tensor& sub_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto tmp = sub_tensor(self, other, alpha);
  ::tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor sub_scalar(
    const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  // self - alpha * other  ==  self + (-alpha * other)
  return add_scalar(self, other, -alpha.toDouble());
}

}  // namespace tvarant::ops

namespace {

at::Tensor wrap_zeros(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  return tvarant::ops::zeros(size, dtype, layout, device, pin_memory);
}

at::Tensor wrap_zeros_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  return tvarant::ops::zeros_like(self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_ones(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  return tvarant::ops::ones(size, dtype, layout, device, pin_memory);
}

at::Tensor wrap_ones_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  return tvarant::ops::ones_like(self, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_full(
    c10::IntArrayRef size,
    const at::Scalar& fill_value,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  return tvarant::ops::full(size, fill_value, dtype, layout, device, pin_memory);
}

at::Tensor wrap_full_like(
    const at::Tensor& self,
    const at::Scalar& fill_value,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  return tvarant::ops::full_like(
      self, fill_value, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_eye(
    int64_t n,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  return tvarant::ops::eye(n, dtype, layout, device, pin_memory);
}

at::Tensor wrap_eye_m(
    int64_t n,
    int64_t m,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  return tvarant::ops::eye_m(n, m, dtype, layout, device, pin_memory);
}

at::Tensor wrap_clone(const at::Tensor& self, std::optional<c10::MemoryFormat> memory_format) {
  return tvarant::ops::clone(self, memory_format);
}

at::Tensor wrap_contiguous(const at::Tensor& self, c10::MemoryFormat memory_format) {
  return tvarant::ops::contiguous(self, memory_format);
}

int g_tvarant_extended_force_link = 0;

}  // namespace

int tvarant_extended_force_link() {
  return g_tvarant_extended_force_link;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("zeros", TORCH_FN(wrap_zeros));
  m.impl("zeros_like", TORCH_FN(wrap_zeros_like));
  m.impl("ones", TORCH_FN(wrap_ones));
  m.impl("ones_like", TORCH_FN(wrap_ones_like));
  m.impl("full", TORCH_FN(wrap_full));
  m.impl("full_like", TORCH_FN(wrap_full_like));
  m.impl("eye", TORCH_FN(wrap_eye));
  m.impl("eye.m", TORCH_FN(wrap_eye_m));
  m.impl("clone", TORCH_FN(wrap_clone));
  m.impl("contiguous", TORCH_FN(wrap_contiguous));
  m.impl("neg", TORCH_FN(tvarant::ops::neg));
  m.impl("neg_", TORCH_FN(tvarant::ops::neg_));
  m.impl("neg.out", TORCH_FN(tvarant::ops::neg_out));
  m.impl("abs", TORCH_FN(tvarant::ops::abs));
  m.impl("abs_", TORCH_FN(tvarant::ops::abs_));
  m.impl("abs.out", TORCH_FN(tvarant::ops::abs_out));
  m.impl("exp", TORCH_FN(tvarant::ops::exp));
  m.impl("exp_", TORCH_FN(tvarant::ops::exp_));
  m.impl("exp.out", TORCH_FN(tvarant::ops::exp_out));
  m.impl("exp2", TORCH_FN(tvarant::ops::exp2));
  m.impl("exp2_", TORCH_FN(tvarant::ops::exp2_));
  m.impl("exp2.out", TORCH_FN(tvarant::ops::exp2_out));
  m.impl("sqrt", TORCH_FN(tvarant::ops::sqrt));
  m.impl("sqrt_", TORCH_FN(tvarant::ops::sqrt_));
  m.impl("sqrt.out", TORCH_FN(tvarant::ops::sqrt_out));
  m.impl("rsqrt", TORCH_FN(tvarant::ops::rsqrt));
  m.impl("rsqrt_", TORCH_FN(tvarant::ops::rsqrt_));
  m.impl("rsqrt.out", TORCH_FN(tvarant::ops::rsqrt_out));
  m.impl("sub.Tensor", TORCH_FN(tvarant::ops::sub_tensor));
  m.impl("sub.out", TORCH_FN(tvarant::ops::sub_out));
  m.impl("sub.Scalar", TORCH_FN(tvarant::ops::sub_scalar));
}
