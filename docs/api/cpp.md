# C++ extension

`torch_tvarant` is a PyTorch C++ extension (`PrivateUse1`) built from `csrc/`.

## Layout

| Path | Role |
|---|---|
| `csrc/Module.cpp` | pybind entry: device helpers, force-link registrations |
| `csrc/aten/` | ATen `TORCH_LIBRARY_IMPL(aten, PrivateUse1, …)` |
| `csrc/runtime/` | CPU sim + OpenCL runtimes |
| `csrc/kernels/host/` | Host fp32 kernels |
| `csrc/kernels/opencl/` | OpenCL `.cl` sources |
| `csrc/jit/` | Pointwise / GEMM fusion helpers |

## Registering an ATen op

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("relu", TORCH_FN(tvarant::ops::relu));
}
```

Implement the kernel (host + optional OpenCL), wire `runtime().launch`, then
register the impl in `Ops.cpp`, `LlmOps.cpp`, or another `*Ops.cpp` TU that is
force-linked from `Module.cpp`.

Qualify calls into `tvarant::ops::…` when names collide with ATen ADL
(see existing `silu` / unary helpers).

## Custom library

`TORCH_LIBRARY(tvarant, m)` in `CustomOps.cpp` defines
`torch.ops.tvarant.linear_act` and `torch.ops.tvarant.pointwise`.

## Build

```bash
pip install -e . --no-build-isolation
# OpenCL:
USE_OPENCL=1 pip install -e . --no-build-isolation
```

See [Installation](../install.md) and [Contributing](../contributing.md).
