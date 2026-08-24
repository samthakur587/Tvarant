#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace tvarant::jit {

// SSA pointwise program. The last node is the stored output.
enum class PwOp : uint8_t {
  Load = 0,
  Const = 1,
  Add = 2,      // a + alpha * b
  Mul = 3,      // a * b
  Relu = 4,     // max(a, 0)
  Silu = 5,     // a * sigmoid(a)
  Neg = 6,      // -a
  Scale = 7,    // alpha * a
  Sigmoid = 8,  // 1 / (1 + exp(-a))
};

struct PwNode {
  PwOp op{PwOp::Load};
  int a{-1};
  int b{-1};
  int input_id{-1};
  float alpha{1.f};
  float c{0.f};
};

struct PointwiseProgram {
  std::vector<PwNode> nodes;
  int n_inputs{0};

  std::string cache_key() const;
};

void run_pointwise_host(
    const PointwiseProgram& prog,
    const float* const* inputs,
    float* out,
    int64_t n);

// OpenCL C source for a fused kernel named `kernel_name`.
std::string codegen_opencl(const PointwiseProgram& prog, const std::string& kernel_name);

inline float apply_act(float x, int act) {
  if (act == 1) {
    return x > 0.f ? x : 0.f;
  }
  if (act == 2) {
    return x / (1.f + std::exp(-x));
  }
  return x;
}

int parse_act(const std::string& name);

}  // namespace tvarant::jit
