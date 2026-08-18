#include "TvarantRuntime.h"

#include <c10/util/Exception.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

namespace tvarant {

std::unique_ptr<TvarantRuntime> make_cpu_sim_runtime();
std::unique_ptr<TvarantRuntime> make_opencl_runtime();

namespace {

std::unique_ptr<TvarantRuntime> g_runtime;
std::once_flag g_once;
thread_local int g_current_device = 0;

Backend parse_backend() {
  const char* env = std::getenv("TVARANT_BACKEND");
  if (env == nullptr || env[0] == '\0' || std::string(env) == "sim") {
    return Backend::Sim;
  }
  if (std::string(env) == "opencl" || std::string(env) == "fpga") {
    return Backend::OpenCL;
  }
  TORCH_CHECK(
      false,
      "Unknown TVARANT_BACKEND=",
      env,
      " (expected 'sim' or 'opencl')");
  return Backend::Sim;
}

}  // namespace

void initialize_runtime() {
  std::call_once(g_once, []() {
    if (parse_backend() == Backend::OpenCL) {
      g_runtime = make_opencl_runtime();
    } else {
      g_runtime = make_cpu_sim_runtime();
    }
  });
}

TvarantRuntime& runtime() {
  initialize_runtime();
  TORCH_CHECK(g_runtime != nullptr, "Tvarant runtime is not initialized");
  return *g_runtime;
}

int current_device() {
  return g_current_device;
}

void set_device(int index) {
  TORCH_CHECK(
      index >= 0 && index < kDeviceCount,
      "Tvarant device index out of range: ",
      index);
  g_current_device = index;
}

int exchange_device(int index) {
  int prev = g_current_device;
  set_device(index);
  return prev;
}

}  // namespace tvarant
