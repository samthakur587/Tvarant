# API reference

PyTorch-style reference for **torch_tvarant**. Each entry is a dedicated page
with signature, parameters, and examples.

## torch.tvarant

Device module — [`torch.tvarant` overview](torch/index.md)

- [`is_available`](torch/is_available.md) · [`device_count`](torch/device_count.md) · [`backend`](torch/backend.md)
- [`set_device`](torch/set_device.md) · [`synchronize`](torch/synchronize.md)
- [`manual_seed`](torch/manual_seed.md) · [`get_rng_state`](torch/get_rng_state.md) · …

## torch_tvarant.compiler

- [`compile`](compiler/compile.md) · [`compile_fx`](compiler/compile_fx.md)
- [`trace_module`](compiler/trace_module.md) · [`register`](compiler/register.md)
- [Compiler overview](compiler/index.md)

## Supported ops

- [Ops index](ops/index.md) — `relu`, `add`, `mm`, `linear`, `linear_act`, …
- [Kernels](kernels.md) — host / OpenCL catalog
- [C++](cpp.md) — extension layout & `TORCH_LIBRARY_IMPL`
