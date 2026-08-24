#include "Jit.h"

#include <c10/util/Exception.h>

#include <cmath>
#include <sstream>
#include <string>

namespace tvarant::jit {

std::string PointwiseProgram::cache_key() const {
  std::string s;
  s += std::to_string(n_inputs);
  s.push_back('|');
  s.reserve(s.size() + nodes.size() * 24);
  for (const auto& nd : nodes) {
    s += std::to_string(static_cast<int>(nd.op));
    s.push_back(',');
    s += std::to_string(nd.a);
    s.push_back(',');
    s += std::to_string(nd.b);
    s.push_back(',');
    s += std::to_string(nd.input_id);
    s.push_back(',');
    s += std::to_string(nd.alpha);
    s.push_back(',');
    s += std::to_string(nd.c);
    s.push_back(';');
  }
  return s;
}

void run_pointwise_host(
    const PointwiseProgram& prog,
    const float* const* inputs,
    float* out,
    int64_t n) {
  TORCH_CHECK(!prog.nodes.empty(), "empty pointwise program");
  TORCH_CHECK(prog.n_inputs >= 0, "invalid pointwise input count");
  std::vector<float> regs(prog.nodes.size());
  for (int64_t i = 0; i < n; ++i) {
    for (size_t v = 0; v < prog.nodes.size(); ++v) {
      const auto& nd = prog.nodes[v];
      switch (nd.op) {
        case PwOp::Load:
          TORCH_CHECK(
              nd.input_id >= 0 && nd.input_id < prog.n_inputs, "Load input_id out of range");
          regs[v] = inputs[nd.input_id][i];
          break;
        case PwOp::Const:
          regs[v] = nd.c;
          break;
        case PwOp::Add:
          regs[v] = regs[static_cast<size_t>(nd.a)] + nd.alpha * regs[static_cast<size_t>(nd.b)];
          break;
        case PwOp::Mul:
          regs[v] = regs[static_cast<size_t>(nd.a)] * regs[static_cast<size_t>(nd.b)];
          break;
        case PwOp::Relu: {
          const float x = regs[static_cast<size_t>(nd.a)];
          regs[v] = x > 0.f ? x : 0.f;
          break;
        }
        case PwOp::Silu: {
          const float x = regs[static_cast<size_t>(nd.a)];
          regs[v] = x / (1.f + std::exp(-x));
          break;
        }
        case PwOp::Neg:
          regs[v] = -regs[static_cast<size_t>(nd.a)];
          break;
        case PwOp::Scale:
          regs[v] = nd.alpha * regs[static_cast<size_t>(nd.a)];
          break;
        case PwOp::Sigmoid: {
          const float x = regs[static_cast<size_t>(nd.a)];
          regs[v] = 1.f / (1.f + std::exp(-x));
          break;
        }
        default:
          TORCH_CHECK(false, "unknown pointwise op ", static_cast<int>(nd.op));
      }
    }
    out[i] = regs.back();
  }
}

std::string codegen_opencl(const PointwiseProgram& prog, const std::string& kernel_name) {
  TORCH_CHECK(!prog.nodes.empty(), "empty pointwise program");
  std::ostringstream oss;
  oss << "__kernel void " << kernel_name << "(";
  for (int i = 0; i < prog.n_inputs; ++i) {
    oss << "__global const float* in" << i << ", ";
  }
  oss << "__global float* out, const int n) {\n";
  oss << "  const int i = get_global_id(0);\n";
  oss << "  if (i >= n) return;\n";
  for (size_t v = 0; v < prog.nodes.size(); ++v) {
    const auto& nd = prog.nodes[v];
    oss << "  const float v" << v << " = ";
    switch (nd.op) {
      case PwOp::Load:
        oss << "in" << nd.input_id << "[i]";
        break;
      case PwOp::Const:
        oss << nd.c << "f";
        break;
      case PwOp::Add:
        oss << "v" << nd.a << " + " << nd.alpha << "f * v" << nd.b;
        break;
      case PwOp::Mul:
        oss << "v" << nd.a << " * v" << nd.b;
        break;
      case PwOp::Relu:
        oss << "(v" << nd.a << " > 0.f ? v" << nd.a << " : 0.f)";
        break;
      case PwOp::Silu:
        oss << "(v" << nd.a << " / (1.f + exp(-v" << nd.a << ")))";
        break;
      case PwOp::Neg:
        oss << "(-v" << nd.a << ")";
        break;
      case PwOp::Scale:
        oss << "(" << nd.alpha << "f * v" << nd.a << ")";
        break;
      case PwOp::Sigmoid:
        oss << "(1.f / (1.f + exp(-v" << nd.a << ")))";
        break;
      default:
        TORCH_CHECK(false, "unknown pointwise op in codegen");
    }
    oss << ";\n";
  }
  oss << "  out[i] = v" << (prog.nodes.size() - 1) << ";\n";
  oss << "}\n";
  return oss.str();
}

int parse_act(const std::string& name) {
  if (name.empty() || name == "none" || name == "identity") {
    return 0;
  }
  if (name == "relu") {
    return 1;
  }
  if (name == "silu" || name == "swish") {
    return 2;
  }
  TORCH_CHECK(false, "unknown activation '", name, "' (expected none|relu|silu)");
  return 0;
}

}  // namespace tvarant::jit
