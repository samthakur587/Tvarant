#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tvarant {

constexpr int64_t kLocalWorkSize = 32;
constexpr int kDeviceCount = 1;

enum class Backend { Sim, OpenCL };

struct LaunchParams {
  const char* kernel = nullptr;
  void* dst = nullptr;
  const void* src0 = nullptr;
  const void* src1 = nullptr;
  int64_t numel = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  float alpha = 1.f;
  float beta = 0.f;
  float scalar = 0.f;
};

class TvarantRuntime {
 public:
  virtual ~TvarantRuntime() = default;

  virtual Backend backend() const = 0;
  virtual const char* name() const = 0;
  virtual bool is_host_accessible() const = 0;

  virtual void* alloc(size_t nbytes) = 0;
  virtual void free(void* ptr) = 0;
  virtual bool owns(const void* ptr) const = 0;

  virtual void copy_h2d(void* dst, const void* src, size_t n) = 0;
  virtual void copy_d2h(void* dst, const void* src, size_t n) = 0;
  virtual void copy_d2d(void* dst, const void* src, size_t n) = 0;

  virtual void launch(const LaunchParams& params) = 0;
  virtual void synchronize() = 0;
};

void initialize_runtime();
TvarantRuntime& runtime();

int current_device();
void set_device(int index);
int exchange_device(int index);

inline int64_t round_up_lws(int64_t n) {
  return ((n + kLocalWorkSize - 1) / kLocalWorkSize) * kLocalWorkSize;
}

}  // namespace tvarant
