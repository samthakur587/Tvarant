#include "aten/AtenCommon.h"
#include "jit/Jit.h"
#include "runtime/TvarantRuntime.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <optional>
#include <string>
#include <vector>

namespace tvarant::ops {

at::Tensor linear_act(
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    std::string act,
    bool trans_b) {
  check_fp32(x, "linear_act");
  check_fp32(weight, "linear_act");
  const int act_id = jit::parse_act(act);
  auto xc = x.contiguous();
  TORCH_CHECK(xc.dim() >= 1, "linear_act: expected at least 1D input");
  const int64_t k = xc.size(-1);
  const int64_t m = xc.numel() / k;
  auto x2 = xc.reshape({m, k});
  const at::Tensor* bptr = nullptr;
  at::Tensor bstore;
  if (bias.has_value() && bias->defined()) {
    check_fp32(*bias, "linear_act");
    bstore = *bias;
    bptr = &bstore;
  }
  auto y2 = gemm_bias_act(x2, weight, bptr, 1.f, 1.f, act_id, trans_b);
  auto out_sizes = xc.sizes().vec();
  out_sizes.back() = y2.size(1);
  return y2.reshape(out_sizes);
}

at::Tensor pointwise(
    at::TensorList inputs,
    at::IntArrayRef ops,
    at::IntArrayRef a,
    at::IntArrayRef b,
    at::IntArrayRef input_ids,
    at::ArrayRef<double> alphas,
    at::ArrayRef<double> consts) {
  TORCH_CHECK(!inputs.empty(), "pointwise: need at least one input");
  const auto n_nodes = ops.size();
  TORCH_CHECK(
      a.size() == n_nodes && b.size() == n_nodes && input_ids.size() == n_nodes &&
          alphas.size() == n_nodes && consts.size() == n_nodes,
      "pointwise: SSA arrays must all have the same length");
  TORCH_CHECK(n_nodes > 0, "pointwise: empty program");

  jit::PointwiseProgram prog;
  prog.n_inputs = static_cast<int>(inputs.size());
  prog.nodes.resize(n_nodes);
  for (size_t i = 0; i < n_nodes; ++i) {
    auto& nd = prog.nodes[i];
    nd.op = static_cast<jit::PwOp>(ops[static_cast<int64_t>(i)]);
    nd.a = static_cast<int>(a[static_cast<int64_t>(i)]);
    nd.b = static_cast<int>(b[static_cast<int64_t>(i)]);
    nd.input_id = static_cast<int>(input_ids[static_cast<int64_t>(i)]);
    nd.alpha = static_cast<float>(alphas[static_cast<int64_t>(i)]);
    nd.c = static_cast<float>(consts[static_cast<int64_t>(i)]);
  }

  std::vector<at::Tensor> contig;
  contig.reserve(inputs.size());
  std::vector<const void*> ptrs;
  ptrs.reserve(inputs.size());
  const int64_t numel = inputs[0].numel();
  const auto sizes = inputs[0].sizes();
  const auto options = inputs[0].options();
  for (const auto& t : inputs) {
    check_fp32(t, "pointwise");
    TORCH_CHECK(t.device().is_privateuseone(), "pointwise: expected tvarant tensors");
    auto c = t.contiguous();
    TORCH_CHECK(c.numel() == numel, "pointwise: inputs must have the same numel");
    ptrs.push_back(c.data_ptr());
    contig.push_back(std::move(c));
  }
  auto out = at::empty(sizes, options);
  ::tvarant::runtime().launch_pointwise(
      prog, ptrs.data(), prog.n_inputs, out.data_ptr(), numel);
  return out;
}

}  // namespace tvarant::ops

namespace {
int g_tvarant_custom_force_link = 0;
}  // namespace

int tvarant_custom_force_link() {
  return g_tvarant_custom_force_link;
}

TORCH_LIBRARY(tvarant, m) {
  m.def(
      "linear_act(Tensor x, Tensor weight, Tensor? bias, str act, bool trans_b=False) -> Tensor");
  m.def(
      "pointwise(Tensor[] inputs, int[] ops, int[] a, int[] b, int[] input_ids, float[] alphas, float[] consts) -> Tensor");
}

TORCH_LIBRARY_IMPL(tvarant, PrivateUse1, m) {
  m.impl("linear_act", TORCH_FN(tvarant::ops::linear_act));
  m.impl("pointwise", TORCH_FN(tvarant::ops::pointwise));
}
