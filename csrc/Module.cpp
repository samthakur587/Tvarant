#include "runtime/TvarantRuntime.h"

#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/extension.h>

namespace tvarant {
int force_link_allocator();
int force_link_guard();
int force_link_hooks();
int force_link_generator();
const at::Generator& get_default_generator(c10::DeviceIndex device_index);
}  // namespace tvarant

int tvarant_aten_force_link();
int tvarant_llm_force_link();
int tvarant_custom_force_link();
int tvarant_extended_force_link();

namespace {

int force_link_all() {
  return tvarant::force_link_allocator() + tvarant::force_link_guard() +
      tvarant::force_link_hooks() + tvarant::force_link_generator() +
      tvarant_aten_force_link() + tvarant_llm_force_link() +
      tvarant_custom_force_link() + tvarant_extended_force_link();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  (void)force_link_all();
  tvarant::initialize_runtime();
  c10::register_privateuse1_backend("tvarant");

  m.def("is_available", []() { return true; });
  m.def("device_count", []() { return tvarant::kDeviceCount; });
  m.def("current_device", []() { return tvarant::current_device(); });
  m.def("set_device", [](int index) { tvarant::set_device(index); });
  m.def("synchronize", []() { tvarant::runtime().synchronize(); });
  m.def("backend", []() { return std::string(tvarant::runtime().name()); });
  m.def("device", []() {
    return c10::Device(c10::DeviceType::PrivateUse1, tvarant::current_device());
  });
  m.def("manual_seed", [](int64_t seed) {
    auto gen = tvarant::get_default_generator(0);
    gen.set_current_seed(static_cast<uint64_t>(seed));
  });
  m.def("get_rng_state", []() {
    return tvarant::get_default_generator(0).get_state();
  });
  m.def("set_rng_state", [](const at::Tensor& state) {
    auto gen = tvarant::get_default_generator(0);
    gen.set_state(state);
  });
}
