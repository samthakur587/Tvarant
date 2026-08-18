#include "TvarantRuntime.h"

#include <ATen/CPUGeneratorImpl.h>

namespace tvarant {

class TvarantGeneratorImpl final : public at::CPUGeneratorImpl {
 public:
  explicit TvarantGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
};

const at::Generator& get_default_generator(c10::DeviceIndex device_index) {
  static at::Generator gen = []() {
    auto g = at::make_generator<TvarantGeneratorImpl>(0);
    g.seed();
    return g;
  }();
  if (device_index == -1) {
    device_index = static_cast<c10::DeviceIndex>(current_device());
  }
  TORCH_CHECK(device_index == 0, "Tvarant only supports device index 0");
  return gen;
}

at::Generator make_generator(c10::DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = static_cast<c10::DeviceIndex>(current_device());
  }
  return at::make_generator<TvarantGeneratorImpl>(device_index);
}

}  // namespace tvarant

// Generator registration is via PrivateUse1HooksInterface::{getDefaultGenerator,
// getNewGenerator}. REGISTER_GENERATOR_PRIVATEUSE1 is deprecated in PyTorch 2.11.

namespace tvarant {
int force_link_generator() {
  return 1;
}
}  // namespace tvarant
