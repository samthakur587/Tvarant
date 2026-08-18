#include "TvarantRuntime.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Storage.h>

#include <algorithm>

namespace tvarant {

const at::Generator& get_default_generator(c10::DeviceIndex device_index);
at::Generator make_generator(c10::DeviceIndex device_index);
c10::Allocator* get_tvarant_allocator();

struct TvarantHooks final : public at::PrivateUse1HooksInterface {
  ~TvarantHooks() override = default;

  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override {
    return true;
  }

  bool hasPrimaryContext(c10::DeviceIndex /*device_index*/) const override {
    return true;
  }

  void init() const override {
    initialize_runtime();
  }

  c10::DeviceIndex deviceCount() const override {
    return kDeviceCount;
  }

  void setCurrentDevice(c10::DeviceIndex device) const override {
    set_device(device);
  }

  c10::DeviceIndex getCurrentDevice() const override {
    return static_cast<c10::DeviceIndex>(current_device());
  }

  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
    return static_cast<c10::DeviceIndex>(exchange_device(device));
  }

  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
    if (device < 0 || device >= kDeviceCount) {
      return getCurrentDevice();
    }
    return exchangeDevice(device);
  }

  at::Device getDeviceFromPtr(void* data) const override {
    TORCH_CHECK(runtime().owns(data), "Pointer is not a Tvarant allocation");
    return at::Device(at::DeviceType::PrivateUse1, current_device());
  }

  bool isPinnedPtr(const void* /*data*/) const override {
    return false;
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return c10::GetAllocator(c10::DeviceType::CPU);
  }

  const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override {
    return get_default_generator(device_index);
  }

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return make_generator(device_index);
  }

  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
    auto* impl = storage.unsafeGetStorageImpl();
    TORCH_CHECK(impl->resizable(), "Trying to resize storage that is not resizable");
    if (newsize == 0) {
      impl->set_data_ptr_noswap(get_tvarant_allocator()->allocate(0));
      impl->set_nbytes(0);
      return;
    }
    at::DataPtr new_data = get_tvarant_allocator()->allocate(newsize);
    const size_t old_nbytes = impl->nbytes();
    if (old_nbytes > 0 && impl->data()) {
      runtime().copy_d2d(
          new_data.get(), impl->data(), std::min(old_nbytes, newsize));
    }
    impl->set_data_ptr_noswap(std::move(new_data));
    impl->set_nbytes(newsize);
  }
};

}  // namespace tvarant

namespace {

static bool g_hooks_registered [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new tvarant::TvarantHooks());
  return true;
}();

}  // namespace

namespace tvarant {
int force_link_hooks() {
  return 1;
}
}  // namespace tvarant
