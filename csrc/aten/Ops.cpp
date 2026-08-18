#include "runtime/TvarantRuntime.h"

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace tvarant::ops {
namespace {

c10::DispatchKeySet tvarant_keys() {
  return c10::DispatchKeySet{
      c10::DispatchKey::PrivateUse1, c10::DispatchKey::AutogradPrivateUse1};
}

c10::Allocator* allocator() {
  return at::GetAllocator(at::kPrivateUse1);
}

void check_fp32(const at::Tensor& t, const char* op) {
  TORCH_CHECK(
      t.scalar_type() == at::kFloat,
      "Tvarant op '",
      op,
      "' supports float32 only, got ",
      t.scalar_type());
}

at::Tensor as_cpu_view(const at::Tensor& t) {
  TORCH_CHECK(
      ::tvarant::runtime().is_host_accessible(),
      "as_cpu_view requires a host-accessible Tvarant runtime");
  return at::from_blob(
      t.data_ptr(),
      t.sizes(),
      t.strides(),
      t.options().device(at::kCPU).dtype(t.dtype()));
}

at::Tensor storage_to_cpu(const at::Tensor& t) {
  if (t.is_cpu()) {
    return t;
  }
  const auto nbytes = t.storage().nbytes();
  const int64_t n_items =
      static_cast<int64_t>(nbytes / std::max<size_t>(t.dtype().itemsize(), 1));
  auto cpu_buf = at::empty({n_items}, t.options().device(at::kCPU).dtype(t.dtype()));
  if (nbytes > 0) {
    ::tvarant::runtime().copy_d2h(cpu_buf.data_ptr(), t.storage().data(), nbytes);
  }
  auto blob = at::from_blob(
      static_cast<char*>(cpu_buf.data_ptr()) +
          t.storage_offset() * static_cast<int64_t>(t.dtype().itemsize()),
      t.sizes(),
      t.strides(),
      t.options().device(at::kCPU).dtype(t.dtype()));
  return blob.clone();
}

void cpu_into_device(const at::Tensor& cpu, const at::Tensor& dst) {
  TORCH_CHECK(dst.device().is_privateuseone());
  auto cpu_c = cpu.to(dst.dtype());
  if (dst.is_contiguous() && cpu_c.sizes() == dst.sizes()) {
    auto src = cpu_c.contiguous();
    ::tvarant::runtime().copy_h2d(
        dst.data_ptr(),
        src.data_ptr(),
        static_cast<size_t>(dst.numel()) * dst.dtype().itemsize());
    return;
  }
  auto dst_cpu = storage_to_cpu(dst);
  dst_cpu.copy_(cpu_c);
  const auto nbytes = dst.storage().nbytes();
  if (nbytes == 0) {
    return;
  }
  auto packed = at::empty(
      {static_cast<int64_t>(nbytes / dst.dtype().itemsize())},
      dst.options().device(at::kCPU).dtype(dst.dtype()));
  auto view = at::from_blob(
      static_cast<char*>(packed.data_ptr()) +
          dst.storage_offset() * static_cast<int64_t>(dst.dtype().itemsize()),
      dst.sizes(),
      dst.strides(),
      packed.options());
  view.copy_(dst_cpu);
  ::tvarant::runtime().copy_h2d(dst.storage().mutable_data(), packed.data_ptr(), nbytes);
}

}  // namespace

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone(), "empty(): expected tvarant device");
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Tvarant only supports strided layout");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory is only supported on CPU");
  const c10::DeviceGuard guard(device);
  return at::detail::empty_generic(
      size, allocator(), tvarant_keys(), dtype, memory_format_opt);
}

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone(), "empty_strided(): expected tvarant device");
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Tvarant only supports strided layout");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory is only supported on CPU");
  const c10::DeviceGuard guard(device);
  return at::detail::empty_strided_generic(
      size, stride, allocator(), tvarant_keys(), dtype);
}

at::Tensor as_strided(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {
  const int64_t offset = storage_offset.value_or(self.storage_offset());
  auto result = at::detail::make_tensor<c10::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(self.storage()),
      self.key_set(),
      self.dtype());
  at::native::setStrided(result, size, stride, offset);
  return result;
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::IntArrayRef size,
    std::optional<c10::MemoryFormat> memory_format) {
  auto* impl = self.unsafeGetTensorImpl();
  impl->set_sizes_contiguous(size);
  const auto itemsize = self.dtype().itemsize();
  size_t nbytes = at::detail::computeStorageNbytesContiguous(size, itemsize);
  nbytes += static_cast<size_t>(self.storage_offset()) * itemsize;
  if (!self.storage() || self.storage().nbytes() < nbytes) {
    at::detail::getPrivateUse1Hooks().resizePrivateUse1Bytes(self.storage(), nbytes);
  }
  if (memory_format.has_value()) {
    impl->empty_tensor_restride(*memory_format);
  }
  return self;
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return tvarant::ops::as_strided(self, size, stride, self.storage_offset());
}

at::Tensor view(const at::Tensor& self, c10::IntArrayRef size) {
  auto inferred = at::infer_size(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred);
  TORCH_CHECK(
      stride.has_value(),
      "view size is not compatible with input tensor's size and stride; use reshape() instead");
  return tvarant::ops::as_strided(self, inferred, *stride, self.storage_offset());
}

at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  (void)non_blocking;
  TORCH_CHECK(self.defined() && dst.defined(), "copy: undefined tensor");
  if (self.numel() == 0) {
    return dst;
  }

  if (::tvarant::runtime().is_host_accessible()) {
    at::Tensor dst_cpu = dst.is_cpu() ? dst : as_cpu_view(dst);
    at::Tensor src_cpu = self.is_cpu() ? self : as_cpu_view(self);
    dst_cpu.copy_(src_cpu);
    return dst;
  }

  if (self.is_cpu() && dst.device().is_privateuseone()) {
    cpu_into_device(self, dst);
    return dst;
  }
  if (self.device().is_privateuseone() && dst.is_cpu()) {
    dst.copy_(storage_to_cpu(self));
    return dst;
  }
  if (self.device().is_privateuseone() && dst.device().is_privateuseone()) {
    cpu_into_device(storage_to_cpu(self), dst);
    return dst;
  }
  TORCH_CHECK(false, "Unsupported Tvarant copy: ", self.device(), " -> ", dst.device());
}

at::Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  tvarant::ops::resize_(dst, self.sizes(), std::nullopt);
  return tvarant::ops::_copy_from(self, dst, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  TORCH_CHECK(self.numel() == 1, "item() only supports tensors with one element");
  auto cpu = at::empty({}, self.options().device(at::kCPU).dtype(self.dtype()));
  ::tvarant::runtime().copy_d2h(
      cpu.data_ptr(), self.data_ptr(), self.dtype().itemsize());
  return cpu.item();
}

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  auto* impl = self.unsafeGetTensorImpl();
  impl->set_storage_keep_dtype(source.storage());
  at::native::setStrided(
      self, source.sizes(), source.strides(), source.storage_offset());
  return self;
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  const int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  auto* impl = self.unsafeGetTensorImpl();
  impl->set_storage_keep_dtype(std::move(source));
  at::native::setStrided(self, {new_size}, {1}, static_cast<int64_t>(0));
  return self;
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
  at::native::setStrided(result, size, stride, storage_offset);
  return result;
}

bool has_compatible_shallow_copy_type(
    const at::Tensor& /*self*/, const at::Tensor& /*other*/) {
  return true;
}

at::Tensor& fill__scalar(at::Tensor& self, const at::Scalar& value) {
  if (self.numel() == 0) {
    return self;
  }
  if (self.scalar_type() == at::kFloat && self.is_contiguous()) {
    ::tvarant::LaunchParams p;
    p.kernel = "fill_kernel";
    p.dst = self.data_ptr();
    p.scalar = value.toFloat();
    p.numel = self.numel();
    ::tvarant::runtime().launch(p);
    return self;
  }
  if (::tvarant::runtime().is_host_accessible()) {
    as_cpu_view(self).fill_(value);
    return self;
  }
  auto cpu = storage_to_cpu(self);
  cpu.fill_(value);
  cpu_into_device(cpu, self);
  return self;
}

at::Tensor add_tensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  check_fp32(self, "add");
  check_fp32(other, "add");
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  auto a = self.expand(out_sizes).contiguous();
  auto b = other.expand(out_sizes).contiguous();
  auto out = at::empty(out_sizes, self.options());
  ::tvarant::LaunchParams p;
  p.kernel = "add_kernel";
  p.src0 = a.data_ptr();
  p.src1 = b.data_ptr();
  p.dst = out.data_ptr();
  p.alpha = alpha.toFloat();
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto tmp = add_tensor(self, other, alpha);
  tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor mul_tensor(const at::Tensor& self, const at::Tensor& other) {
  check_fp32(self, "mul");
  check_fp32(other, "mul");
  auto out_sizes = at::infer_size(self.sizes(), other.sizes());
  auto a = self.expand(out_sizes).contiguous();
  auto b = other.expand(out_sizes).contiguous();
  auto out = at::empty(out_sizes, self.options());
  ::tvarant::LaunchParams p;
  p.kernel = "mul_kernel";
  p.src0 = a.data_ptr();
  p.src1 = b.data_ptr();
  p.dst = out.data_ptr();
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor& mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  auto tmp = mul_tensor(self, other);
  tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor relu(const at::Tensor& self) {
  check_fp32(self, "relu");
  auto in = self.contiguous();
  auto out = at::empty(in.sizes(), in.options());
  ::tvarant::LaunchParams p;
  p.kernel = "relu_kernel";
  p.src0 = in.data_ptr();
  p.dst = out.data_ptr();
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor& relu_(at::Tensor& self) {
  auto out = tvarant::ops::relu(self);
  tvarant::ops::_copy_from(out, self, false);
  return self;
}

at::Tensor& relu_out(const at::Tensor& self, at::Tensor& out) {
  auto tmp = tvarant::ops::relu(self);
  tvarant::ops::_copy_from(tmp, out, false);
  return out;
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  check_fp32(self, "mm");
  check_fp32(mat2, "mm");
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "mm(): tensors must be 2D");
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "mm(): size mismatch ",
      self.sizes(),
      " and ",
      mat2.sizes());
  auto a = self.contiguous();
  auto b = mat2.contiguous();
  auto out = at::empty({a.size(0), b.size(1)}, a.options());
  ::tvarant::LaunchParams p;
  p.kernel = "gemm_kernel";
  p.src0 = a.data_ptr();
  p.src1 = b.data_ptr();
  p.dst = out.data_ptr();
  p.m = a.size(0);
  p.n = b.size(1);
  p.k = a.size(1);
  p.alpha = 1.f;
  p.beta = 0.f;
  p.numel = out.numel();
  ::tvarant::runtime().launch(p);
  return out;
}

at::Tensor addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  check_fp32(mat1, "addmm");
  check_fp32(mat2, "addmm");
  auto out = tvarant::ops::mm(mat1, mat2);
  if (alpha.toFloat() != 1.f) {
    out = tvarant::ops::mul_tensor(out, at::full({}, alpha, out.options()));
  }
  if (beta.toFloat() == 0.f) {
    return out;
  }
  auto bias = self.to(out.dtype());
  if (bias.dim() == 1) {
    bias = bias.unsqueeze(0).expand_as(out);
  } else {
    bias = bias.expand_as(out);
  }
  if (beta.toFloat() != 1.f) {
    bias = tvarant::ops::mul_tensor(bias.contiguous(), at::full({}, beta, out.options()));
  }
  return tvarant::ops::add_tensor(bias.contiguous(), out, 1);
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

}  // namespace tvarant::ops

namespace {

at::Tensor wrap_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  return tvarant::ops::empty_memory_format(
      size, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrap_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory) {
  return tvarant::ops::empty_strided(size, stride, dtype, layout, device, pin_memory);
}

at::Tensor wrap_as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  std::optional<int64_t> off;
  if (storage_offset.has_value()) {
    off = storage_offset->expect_int();
  }
  return tvarant::ops::as_strided(
      self,
      C10_AS_INTARRAYREF_SLOW(size),
      C10_AS_INTARRAYREF_SLOW(stride),
      off);
}

const at::Tensor& wrap_resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<c10::MemoryFormat> memory_format) {
  return tvarant::ops::resize_(self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor wrap__reshape_alias(
    const at::Tensor& self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  return tvarant::ops::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor wrap_view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return tvarant::ops::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

int g_tvarant_aten_force_link = 0;

}  // namespace

int tvarant_aten_force_link() {
  return g_tvarant_aten_force_link;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(wrap_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(wrap_empty_strided));
  m.impl("as_strided", TORCH_FN(wrap_as_strided));
  m.impl("resize_", TORCH_FN(wrap_resize_));
  m.impl("_reshape_alias", TORCH_FN(wrap__reshape_alias));
  m.impl("view", TORCH_FN(wrap_view));
  m.impl("_copy_from", TORCH_FN(tvarant::ops::_copy_from));
  m.impl("_copy_from_and_resize", TORCH_FN(tvarant::ops::_copy_from_and_resize));
  m.impl("_local_scalar_dense", TORCH_FN(tvarant::ops::_local_scalar_dense));
  m.impl("set_.source_Tensor", TORCH_FN(tvarant::ops::set_source_Tensor_));
  m.impl("set_.source_Storage", TORCH_FN(tvarant::ops::set_source_Storage_));
  m.impl(
      "set_.source_Storage_storage_offset",
      TORCH_FN(tvarant::ops::set_source_Storage_storage_offset_));
  m.impl(
      "_has_compatible_shallow_copy_type",
      TORCH_FN(tvarant::ops::has_compatible_shallow_copy_type));
  m.impl("fill_.Scalar", TORCH_FN(tvarant::ops::fill__scalar));
  m.impl("add.Tensor", TORCH_FN(tvarant::ops::add_tensor));
  m.impl("add.out", TORCH_FN(tvarant::ops::add_out));
  m.impl("mul.Tensor", TORCH_FN(tvarant::ops::mul_tensor));
  m.impl("mul.out", TORCH_FN(tvarant::ops::mul_out));
  m.impl("relu", TORCH_FN(tvarant::ops::relu));
  m.impl("relu_", TORCH_FN(tvarant::ops::relu_));
  m.impl("relu.out", TORCH_FN(tvarant::ops::relu_out));
  m.impl("mm", TORCH_FN(tvarant::ops::mm));
  m.impl("addmm", TORCH_FN(tvarant::ops::addmm));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&tvarant::ops::cpu_fallback>());
}
