#include "TvarantRuntime.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#ifdef USE_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#endif

namespace tvarant {

std::unique_ptr<TvarantRuntime> make_cpu_sim_runtime();

#ifdef USE_OPENCL

namespace {

const char* kEmbeddedKernels = R"CLC(
__kernel void fill_kernel(__global float* out, const float value, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = value;
}
__kernel void copy_kernel(__global float* dst, __global const float* src, const int n) {
  const int i = get_global_id(0);
  if (i < n) dst[i] = src[i];
}
__kernel void add_kernel(__global const float* a, __global const float* b, __global float* out, const float alpha, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = a[i] + alpha * b[i];
}
__kernel void mul_kernel(__global const float* a, __global const float* b, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = a[i] * b[i];
}
__kernel void relu_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = in[i] > 0.f ? in[i] : 0.f;
}
__kernel void silu_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) { const float x = in[i]; out[i] = x / (1.f + exp(-x)); }
}
__kernel void neg_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = -in[i];
}
__kernel void abs_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = fabs(in[i]);
}
__kernel void exp_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = exp(in[i]);
}
__kernel void exp2_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = exp2(in[i]);
}
__kernel void sqrt_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = sqrt(in[i]);
}
__kernel void rsqrt_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = rsqrt(in[i]);
}
__kernel void scale_kernel(__global const float* in, __global float* out, const float scale, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = in[i] * scale;
}
__kernel void add_scalar_kernel(__global const float* in, __global float* out, const float scalar, const int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = in[i] + scalar;
}
__kernel void gemm_kernel(__global const float* a, __global const float* b, __global float* c,
                          const int M, const int N, const int K, const float alpha, const float beta) {
  const int gid = get_global_id(0);
  const int i = gid / N;
  const int j = gid - i * N;
  if (i >= M || j >= N) return;
  float acc = 0.f;
  for (int p = 0; p < K; ++p) acc += a[i * K + p] * b[p * N + j];
  const float prev = (beta == 0.f) ? 0.f : c[i * N + j];
  c[i * N + j] = alpha * acc + beta * prev;
}
__kernel void gemm_bias_act_kernel(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c,
    const int M, const int N, const int K, const float alpha, const float beta,
    const int act, const int trans_b, const int has_bias) {
  const int gid = get_global_id(0);
  const int i = gid / N;
  const int j = gid - i * N;
  if (i >= M || j >= N) return;
  float acc = 0.f;
  for (int p = 0; p < K; ++p) {
    const float bv = trans_b ? b[j * K + p] : b[p * N + j];
    acc += a[i * K + p] * bv;
  }
  float prev = 0.f;
  if (has_bias) prev = bias[j];
  else if (beta != 0.f) prev = c[i * N + j];
  acc = alpha * acc + beta * prev;
  if (act == 1) acc = acc > 0.f ? acc : 0.f;
  else if (act == 2) acc = acc / (1.f + exp(-acc));
  c[i * N + j] = acc;
}
__kernel void bmm_kernel(
    __global const float* a, __global const float* b, __global float* c,
    const int B, const int M, const int N, const int K, const int trans_b) {
  const int gid = get_global_id(0);
  const int total = B * M * N;
  if (gid >= total) return;
  int tmp = gid;
  const int j = tmp % N; tmp /= N;
  const int i = tmp % M; tmp /= M;
  const int bi = tmp;
  const int a_off = bi * M * K;
  const int b_off = trans_b ? bi * N * K : bi * K * N;
  const int c_off = bi * M * N;
  float acc = 0.f;
  for (int p = 0; p < K; ++p) {
    const float bv = trans_b ? b[b_off + j * K + p] : b[b_off + p * N + j];
    acc += a[a_off + i * K + p] * bv;
  }
  c[c_off + i * N + j] = acc;
}
__kernel void softmax_kernel(
    __global const float* in, __global float* out, const int outer, const int dim, const int inner) {
  const int gid = get_global_id(0);
  const int total = outer * inner;
  if (gid >= total) return;
  const int o = gid / inner;
  const int i = gid - o * inner;
  float m = in[(o * dim) * inner + i];
  for (int d = 1; d < dim; ++d) {
    const float v = in[(o * dim + d) * inner + i];
    m = v > m ? v : m;
  }
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float e = exp(in[(o * dim + d) * inner + i] - m);
    out[(o * dim + d) * inner + i] = e;
    sum += e;
  }
  const float inv = 1.f / sum;
  for (int d = 0; d < dim; ++d) out[(o * dim + d) * inner + i] *= inv;
}
__kernel void layernorm_kernel(
    __global const float* in, __global const float* weight, __global const float* bias,
    __global float* out, __global float* mean, __global float* rstd,
    const int outer, const int n, const float eps, const int has_weight, const int has_bias) {
  const int row = get_global_id(0);
  if (row >= outer) return;
  __global const float* src = in + row * n;
  float mu = 0.f;
  for (int j = 0; j < n; ++j) mu += src[j];
  mu /= (float)n;
  float var = 0.f;
  for (int j = 0; j < n; ++j) { const float d = src[j] - mu; var += d * d; }
  var /= (float)n;
  const float rs = 1.f / sqrt(var + eps);
  mean[row] = mu;
  rstd[row] = rs;
  __global float* dst = out + row * n;
  for (int j = 0; j < n; ++j) {
    float y = (src[j] - mu) * rs;
    if (has_weight) y *= weight[j];
    if (has_bias) y += bias[j];
    dst[j] = y;
  }
}
__kernel void embedding_kernel(
    __global const float* weight, __global const int* indices, __global float* out,
    const int n_idx, const int dim, const int vocab) {
  const int gid = get_global_id(0);
  const int row = gid / dim;
  const int col = gid - row * dim;
  if (row >= n_idx) return;
  const int id = indices[row];
  if (id < 0 || id >= vocab) { out[row * dim + col] = 0.f; return; }
  out[row * dim + col] = weight[id * dim + col];
}
)CLC";

#define CL_CHECK(expr)                                                        \
  do {                                                                        \
    cl_int _err = (expr);                                                     \
    TORCH_CHECK(_err == CL_SUCCESS, "OpenCL error ", _err, " at ", #expr);    \
  } while (0)

class OpenCLRuntime final : public TvarantRuntime {
 public:
  OpenCLRuntime() {
    cl_uint nplat = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &nplat));
    TORCH_CHECK(nplat > 0, "No OpenCL platforms found");
    std::vector<cl_platform_id> plats(nplat);
    CL_CHECK(clGetPlatformIDs(nplat, plats.data(), nullptr));

    const char* prefer = std::getenv("TVARANT_OPENCL_PLATFORM");
    cl_platform_id plat = plats[0];
    if (prefer != nullptr) {
      for (auto p : plats) {
        char name[256] = {0};
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        if (std::string(name).find(prefer) != std::string::npos) {
          plat = p;
          break;
        }
      }
    }

    cl_uint ndev = 0;
    cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev);
    TORCH_CHECK(err == CL_SUCCESS && ndev > 0, "No OpenCL devices on the selected platform");
    std::vector<cl_device_id> devs(ndev);
    CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr));
    device_ = devs[0];

    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    CL_CHECK(err);
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    CL_CHECK(err);

    kernel_src_ = load_kernel_source();
    const char* src = kernel_src_.c_str();
    program_ = clCreateProgramWithSource(context_, 1, &src, nullptr, &err);
    CL_CHECK(err);
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t log_size = 0;
      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      std::string log(log_size, '\0');
      clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      TORCH_CHECK(false, "OpenCL program build failed:\n", log);
    }

    auto load = [&](const char* name) {
      cl_int e = CL_SUCCESS;
      cl_kernel k = clCreateKernel(program_, name, &e);
      CL_CHECK(e);
      kernels_[name] = k;
    };
    load("fill_kernel");
    load("copy_kernel");
    load("add_kernel");
    load("mul_kernel");
    load("relu_kernel");
    load("silu_kernel");
    load("neg_kernel");
    load("abs_kernel");
    load("exp_kernel");
    load("exp2_kernel");
    load("sqrt_kernel");
    load("rsqrt_kernel");
    load("scale_kernel");
    load("add_scalar_kernel");
    load("gemm_kernel");
    load("gemm_bias_act_kernel");
    load("bmm_kernel");
    load("softmax_kernel");
    load("layernorm_kernel");
    load("embedding_kernel");
  }

  ~OpenCLRuntime() override {
    synchronize();
    for (auto& kv : jit_kernels_) {
      if (kv.second.kernel) {
        clReleaseKernel(kv.second.kernel);
      }
      if (kv.second.program) {
        clReleaseProgram(kv.second.program);
      }
    }
    for (auto& kv : kernels_) {
      clReleaseKernel(kv.second);
    }
    for (auto& kv : buffers_) {
      clReleaseMemObject(kv.second);
    }
    if (program_) {
      clReleaseProgram(program_);
    }
    if (queue_) {
      clReleaseCommandQueue(queue_);
    }
    if (context_) {
      clReleaseContext(context_);
    }
  }

  Backend backend() const override {
    return Backend::OpenCL;
  }

  const char* name() const override {
    return "opencl";
  }

  bool is_host_accessible() const override {
    return false;
  }

  void* alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_WRITE, nbytes, nullptr, &err);
    CL_CHECK(err);
    void* handle = reinterpret_cast<void*>(buf);
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    buffers_[handle] = buf;
    return handle;
  }

  void free(void* ptr) override {
    if (ptr == nullptr) {
      return;
    }
    cl_mem buf = nullptr;
    {
      std::lock_guard<std::recursive_mutex> lock(mutex_);
      auto it = buffers_.find(ptr);
      if (it == buffers_.end()) {
        return;
      }
      buf = it->second;
      buffers_.erase(it);
    }
    clReleaseMemObject(buf);
  }

  bool owns(const void* ptr) const override {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return buffers_.find(const_cast<void*>(ptr)) != buffers_.end();
  }

  void copy_h2d(void* dst, const void* src, size_t n) override {
    if (n == 0) {
      return;
    }
    CL_CHECK(clEnqueueWriteBuffer(
        queue_, as_mem(dst), CL_TRUE, 0, n, src, 0, nullptr, nullptr));
  }

  void copy_d2h(void* dst, const void* src, size_t n) override {
    if (n == 0) {
      return;
    }
    CL_CHECK(clEnqueueReadBuffer(
        queue_, as_mem(const_cast<void*>(src)), CL_TRUE, 0, n, dst, 0, nullptr, nullptr));
  }

  void copy_d2d(void* dst, const void* src, size_t n) override {
    if (n == 0) {
      return;
    }
    CL_CHECK(clEnqueueCopyBuffer(
        queue_,
        as_mem(const_cast<void*>(src)),
        as_mem(dst),
        0,
        0,
        n,
        0,
        nullptr,
        nullptr));
    CL_CHECK(clFinish(queue_));
  }

  void launch(const LaunchParams& p) override {
    TORCH_CHECK(p.kernel != nullptr, "Tvarant OpenCL launch missing kernel name");
    auto it = kernels_.find(p.kernel);
    TORCH_CHECK(it != kernels_.end(), "Unknown Tvarant OpenCL kernel: ", p.kernel);
    cl_kernel k = it->second;
    int n = static_cast<int>(p.numel);
    int arg = 0;

    auto set_buf = [&](void* ptr) {
      cl_mem mem = as_mem(ptr);
      CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_mem), &mem));
    };
    auto set_float = [&](float v) { CL_CHECK(clSetKernelArg(k, arg++, sizeof(float), &v)); };
    auto set_int = [&](int v) { CL_CHECK(clSetKernelArg(k, arg++, sizeof(int), &v)); };

    const std::string name(p.kernel);
    size_t global = static_cast<size_t>(std::max<int64_t>(round_up_lws(p.numel), kLocalWorkSize));
    if (name == "fill_kernel") {
      set_buf(p.dst);
      set_float(p.scalar);
      set_int(n);
    } else if (name == "copy_kernel") {
      set_buf(p.dst);
      set_buf(const_cast<void*>(p.src0));
      set_int(n);
    } else if (name == "add_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(const_cast<void*>(p.src1));
      set_buf(p.dst);
      set_float(p.alpha);
      set_int(n);
    } else if (name == "mul_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(const_cast<void*>(p.src1));
      set_buf(p.dst);
      set_int(n);
    } else if (name == "relu_kernel" || name == "silu_kernel" || name == "neg_kernel" ||
               name == "abs_kernel" || name == "exp_kernel" || name == "exp2_kernel" ||
               name == "sqrt_kernel" || name == "rsqrt_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(p.dst);
      set_int(n);
    } else if (name == "scale_kernel" || name == "add_scalar_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(p.dst);
      set_float(p.scalar);
      set_int(n);
    } else if (name == "gemm_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(const_cast<void*>(p.src1));
      set_buf(p.dst);
      set_int(static_cast<int>(p.m));
      set_int(static_cast<int>(p.n));
      set_int(static_cast<int>(p.k));
      set_float(p.alpha);
      set_float(p.beta);
      global = static_cast<size_t>(
          std::max<int64_t>(round_up_lws(p.m * p.n), kLocalWorkSize));
    } else if (name == "gemm_bias_act_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(const_cast<void*>(p.src1));
      set_buf(p.src2 != nullptr ? const_cast<void*>(p.src2) : p.dst);
      set_buf(p.dst);
      set_int(static_cast<int>(p.m));
      set_int(static_cast<int>(p.n));
      set_int(static_cast<int>(p.k));
      set_float(p.alpha);
      set_float(p.beta);
      set_int(p.act);
      set_int(p.trans_b ? 1 : 0);
      set_int(p.src2 != nullptr ? 1 : 0);
      global = static_cast<size_t>(
          std::max<int64_t>(round_up_lws(p.m * p.n), kLocalWorkSize));
    } else if (name == "bmm_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(const_cast<void*>(p.src1));
      set_buf(p.dst);
      set_int(static_cast<int>(p.batch));
      set_int(static_cast<int>(p.m));
      set_int(static_cast<int>(p.n));
      set_int(static_cast<int>(p.k));
      set_int(p.trans_b ? 1 : 0);
      global = static_cast<size_t>(
          std::max<int64_t>(round_up_lws(p.batch * p.m * p.n), kLocalWorkSize));
    } else if (name == "softmax_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(p.dst);
      set_int(static_cast<int>(p.m));
      set_int(static_cast<int>(p.n));
      set_int(static_cast<int>(p.k));
      global = static_cast<size_t>(
          std::max<int64_t>(round_up_lws(p.m * p.k), kLocalWorkSize));
    } else if (name == "layernorm_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(p.src1 != nullptr ? const_cast<void*>(p.src1) : p.dst);
      set_buf(p.src2 != nullptr ? const_cast<void*>(p.src2) : p.dst);
      set_buf(p.dst);
      set_buf(p.aux0);
      set_buf(p.aux1);
      set_int(static_cast<int>(p.m));
      set_int(static_cast<int>(p.n));
      set_float(p.scalar);
      set_int(p.src1 != nullptr ? 1 : 0);
      set_int(p.src2 != nullptr ? 1 : 0);
      global = static_cast<size_t>(
          std::max<int64_t>(round_up_lws(p.m), kLocalWorkSize));
    } else if (name == "embedding_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(const_cast<void*>(p.src1));
      set_buf(p.dst);
      set_int(static_cast<int>(p.numel));
      set_int(static_cast<int>(p.n));
      set_int(static_cast<int>(p.k));
      global = static_cast<size_t>(
          std::max<int64_t>(round_up_lws(p.numel * p.n), kLocalWorkSize));
    } else {
      TORCH_CHECK(false, "Unhandled OpenCL kernel: ", name);
    }

    const size_t local = static_cast<size_t>(kLocalWorkSize);
    CL_CHECK(clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue_));
  }

  void launch_pointwise(
      const jit::PointwiseProgram& prog,
      const void* const* inputs,
      int n_inputs,
      void* dst,
      int64_t numel) override {
    TORCH_CHECK(n_inputs == prog.n_inputs, "pointwise input count mismatch");
    TORCH_CHECK(n_inputs <= 8, "pointwise JIT supports at most 8 inputs");
    const std::string key = prog.cache_key();
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    cl_kernel k = nullptr;
    auto it = jit_kernels_.find(key);
    if (it != jit_kernels_.end()) {
      k = it->second.kernel;
    } else {
      const std::string kname = "pw_" + std::to_string(std::hash<std::string>{}(key));
      const std::string src = jit::codegen_opencl(prog, kname);
      cl_int err = CL_SUCCESS;
      const char* csrc = src.c_str();
      cl_program program = clCreateProgramWithSource(context_, 1, &csrc, nullptr, &err);
      CL_CHECK(err);
      err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(program);
        TORCH_CHECK(false, "Tvarant JIT OpenCL build failed:\n", log, "\nsource:\n", src);
      }
      k = clCreateKernel(program, kname.c_str(), &err);
      CL_CHECK(err);
      jit_kernels_[key] = JitKernel{program, k};
    }

    int arg = 0;
    for (int i = 0; i < n_inputs; ++i) {
      cl_mem mem = as_mem(const_cast<void*>(inputs[i]));
      CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_mem), &mem));
    }
    cl_mem out = as_mem(dst);
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(cl_mem), &out));
    int n = static_cast<int>(numel);
    CL_CHECK(clSetKernelArg(k, arg++, sizeof(int), &n));
    size_t global = static_cast<size_t>(std::max<int64_t>(round_up_lws(numel), kLocalWorkSize));
    const size_t local = static_cast<size_t>(kLocalWorkSize);
    CL_CHECK(clEnqueueNDRangeKernel(queue_, k, 1, nullptr, &global, &local, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue_));
  }

  void synchronize() override {
    if (queue_) {
      clFinish(queue_);
    }
  }

 private:
  cl_mem as_mem(void* ptr) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = buffers_.find(ptr);
    TORCH_CHECK(it != buffers_.end(), "Pointer is not a Tvarant OpenCL buffer");
    return it->second;
  }

  static std::string load_kernel_source() {
    const char* dir = std::getenv("TVARANT_KERNEL_DIR");
    if (dir == nullptr || dir[0] == '\0') {
      return std::string(kEmbeddedKernels);
    }
    static const char* files[] = {
        "fill.cl", "copy.cl", "binary.cl", "relu.cl", "elementwise.cl", "gemm.cl", "reduce.cl"};
    std::ostringstream oss;
    for (const char* name : files) {
      std::ifstream in(std::string(dir) + "/" + name);
      TORCH_CHECK(in, "Failed to read Tvarant kernel ", dir, "/", name);
      oss << in.rdbuf() << "\n";
    }
    return oss.str();
  }

  cl_device_id device_{};
  cl_context context_{nullptr};
  cl_command_queue queue_{nullptr};
  cl_program program_{nullptr};
  std::string kernel_src_;
  std::unordered_map<std::string, cl_kernel> kernels_;
  struct JitKernel {
    cl_program program{nullptr};
    cl_kernel kernel{nullptr};
  };
  std::unordered_map<std::string, JitKernel> jit_kernels_;
  std::unordered_map<void*, cl_mem> buffers_;
  mutable std::recursive_mutex mutex_;
};

}  // namespace

std::unique_ptr<TvarantRuntime> make_opencl_runtime() {
  return std::make_unique<OpenCLRuntime>();
}

#else  // !USE_OPENCL

std::unique_ptr<TvarantRuntime> make_opencl_runtime() {
  TORCH_CHECK(
      false,
      "torch_tvarant was built without OpenCL. Rebuild with USE_OPENCL=1 "
      "(and an OpenCL SDK) to use TVARANT_BACKEND=opencl.");
  return nullptr;
}

#endif

}  // namespace tvarant
