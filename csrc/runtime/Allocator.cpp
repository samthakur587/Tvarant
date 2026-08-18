#include "TvarantRuntime.h"

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>

namespace tvarant {
namespace {

void delete_tvarant_ptr(void* ptr) {
  if (ptr != nullptr) {
    runtime().free(ptr);
  }
}

class TvarantAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    if (nbytes > 0) {
      data = runtime().alloc(nbytes);
    }
    const c10::Device device(c10::DeviceType::PrivateUse1, current_device());
    return {data, data, &delete_tvarant_ptr, device};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &delete_tvarant_ptr;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    runtime().copy_d2d(dest, src, count);
  }
};

TvarantAllocator g_allocator;

}  // namespace

c10::Allocator* get_tvarant_allocator() {
  return &g_allocator;
}

int force_link_allocator() {
  return 1;
}

}  // namespace tvarant

namespace {

static c10::AllocatorRegisterer<c10::DeviceType::PrivateUse1> g_register_tvarant_allocator(
    tvarant::get_tvarant_allocator());

}  // namespace
