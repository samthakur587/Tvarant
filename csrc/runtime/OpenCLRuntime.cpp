#include "TvarantRuntime.h"

#include <c10/util/Exception.h>

#include <algorithm>
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
    load("gemm_kernel");
  }

  ~OpenCLRuntime() override {
    synchronize();
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
    std::lock_guard<std::mutex> lock(mutex_);
    buffers_[handle] = buf;
    return handle;
  }

  void free(void* ptr) override {
    if (ptr == nullptr) {
      return;
    }
    cl_mem buf = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
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
    std::lock_guard<std::mutex> lock(mutex_);
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
    } else if (name == "relu_kernel") {
      set_buf(const_cast<void*>(p.src0));
      set_buf(p.dst);
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
    } else {
      TORCH_CHECK(false, "Unhandled OpenCL kernel: ", name);
    }

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
    std::lock_guard<std::mutex> lock(mutex_);
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
        "fill.cl", "copy.cl", "binary.cl", "relu.cl", "gemm.cl"};
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
  std::unordered_map<void*, cl_mem> buffers_;
  mutable std::mutex mutex_;
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
