#include "TvarantRuntime.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace tvarant {

struct TvarantGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  TvarantGuardImpl() = default;
  explicit TvarantGuardImpl(c10::DeviceType t) {
    TORCH_CHECK(t == static_type, "TvarantGuardImpl got non-PrivateUse1 device ", t);
  }

  c10::DeviceType type() const override {
    return static_type;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Expected tvarant device, got ", d);
    const int prev = exchange_device(d.index());
    return c10::Device(static_type, prev);
  }

  c10::Device getDevice() const override {
    return c10::Device(static_type, current_device());
  }

  void setDevice(c10::Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Expected tvarant device, got ", d);
    set_device(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    if (d.is_privateuseone() && d.index() >= 0 && d.index() < kDeviceCount) {
      set_device(d.index());
    }
  }

  c10::Stream getStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getNewStream(c10::Device d, int /*priority*/ = 0) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getStreamFromGlobalPool(
      c10::Device d, bool /*isHighPriority*/ = false) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream exchangeStream(c10::Stream s) const override {
    return s;
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return kDeviceCount;
  }

  bool queryStream(const c10::Stream& /*stream*/) const override {
    return true;
  }

  void synchronizeStream(const c10::Stream& /*stream*/) const override {
    runtime().synchronize();
  }

  void synchronizeDevice(const c10::DeviceIndex /*device_index*/) const override {
    runtime().synchronize();
  }
};

}  // namespace tvarant

C10_REGISTER_GUARD_IMPL(PrivateUse1, tvarant::TvarantGuardImpl);

namespace tvarant {
int force_link_guard() {
  return 1;
}
}  // namespace tvarant
