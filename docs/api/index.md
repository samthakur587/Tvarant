# API reference

This section mirrors a PyTorch-style API layout for **torch_tvarant**.

| Page | Contents |
|---|---|
| [torch.tvarant](torch.md) | Device module helpers + Python package / compiler (auto-generated) |
| [Supported ops](ops.md) | ATen and custom ops registered for `PrivateUse1` |
| [Kernels](kernels.md) | Host and OpenCL kernel catalog |
| [C++](cpp.md) | Extension layout, `TORCH_LIBRARY_IMPL`, adding ops |

Unimplemented ATen ops fall back to CPU; see [Supported ops](ops.md).
