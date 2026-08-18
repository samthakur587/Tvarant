#include "TvarantRuntime.h"
#include "HostKernels.h"

#include <c10/util/Exception.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace tvarant {
namespace {

void* aligned_alloc_bytes(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
#ifdef _WIN32
  void* ptr = _aligned_malloc(nbytes, 64);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 64, nbytes) != 0) {
    ptr = nullptr;
  }
#endif
  TORCH_CHECK(ptr != nullptr, "Tvarant sim allocator failed for ", nbytes, " bytes");
  return ptr;
}

void aligned_free_bytes(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

class CpuSimRuntime final : public TvarantRuntime {
 public:
  Backend backend() const override {
    return Backend::Sim;
  }

  const char* name() const override {
    return "sim";
  }

  bool is_host_accessible() const override {
    return true;
  }

  void* alloc(size_t nbytes) override {
    void* ptr = aligned_alloc_bytes(nbytes);
    if (ptr != nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      live_.insert(ptr);
    }
    return ptr;
  }

  void free(void* ptr) override {
    if (ptr == nullptr) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      live_.erase(ptr);
    }
    aligned_free_bytes(ptr);
  }

  bool owns(const void* ptr) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_.find(const_cast<void*>(ptr)) != live_.end();
  }

  void copy_h2d(void* dst, const void* src, size_t n) override {
    if (n == 0) {
      return;
    }
    std::memcpy(dst, src, n);
  }

  void copy_d2h(void* dst, const void* src, size_t n) override {
    if (n == 0) {
      return;
    }
    std::memcpy(dst, src, n);
  }

  void copy_d2d(void* dst, const void* src, size_t n) override {
    if (n == 0) {
      return;
    }
    std::memcpy(dst, src, n);
  }

  void launch(const LaunchParams& p) override {
    TORCH_CHECK(p.kernel != nullptr, "Tvarant launch missing kernel name");
    const std::string name(p.kernel);
    if (name == "fill_kernel") {
      host::fill_f32(static_cast<float*>(p.dst), p.scalar, p.numel);
    } else if (name == "copy_kernel") {
      host::copy_f32(
          static_cast<float*>(p.dst), static_cast<const float*>(p.src0), p.numel);
    } else if (name == "add_kernel") {
      host::add_f32(
          static_cast<const float*>(p.src0),
          static_cast<const float*>(p.src1),
          static_cast<float*>(p.dst),
          p.numel,
          p.alpha);
    } else if (name == "mul_kernel") {
      host::mul_f32(
          static_cast<const float*>(p.src0),
          static_cast<const float*>(p.src1),
          static_cast<float*>(p.dst),
          p.numel);
    } else if (name == "relu_kernel") {
      host::relu_f32(
          static_cast<const float*>(p.src0), static_cast<float*>(p.dst), p.numel);
    } else if (name == "gemm_kernel") {
      host::gemm_f32(
          static_cast<const float*>(p.src0),
          static_cast<const float*>(p.src1),
          static_cast<float*>(p.dst),
          p.m,
          p.n,
          p.k,
          p.alpha,
          p.beta);
    } else {
      TORCH_CHECK(false, "Unknown Tvarant sim kernel: ", name);
    }
  }

  void synchronize() override {}

 private:
  mutable std::mutex mutex_;
  std::unordered_set<void*> live_;
};

}  // namespace

std::unique_ptr<TvarantRuntime> make_cpu_sim_runtime() {
  return std::make_unique<CpuSimRuntime>();
}

}  // namespace tvarant
