#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <optional>

namespace tvarant::ops {

at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt);

at::Tensor& fill__scalar(at::Tensor& self, const at::Scalar& value);
at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);

at::Tensor add_tensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor add_scalar(
    const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);

}  // namespace tvarant::ops
