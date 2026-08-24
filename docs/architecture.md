# Architecture

torch_tvarant follows the standard PyTorch **PrivateUse1** out-of-tree backend
pattern.

```
┌─────────────────────────────────────────────────────────┐
│  Python: torch_tvarant / torch.device("tvarant")        │
│  torch_tvarant.compiler (FX fusion + torch.compile)     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  ATen dispatch (PrivateUse1)                            │
│  csrc/aten/Ops.cpp, LlmOps.cpp, CustomOps.cpp           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  TvarantRuntime (abstract)                              │
│  csrc/runtime/TvarantRuntime.h                          │
├─────────────────────┬───────────────────────────────────┤
│  CpuSimRuntime      │  OpenCLRuntime                    │
│  (host memory)      │  (cl_mem buffers)                 │
└─────────┬───────────┴───────────────┬───────────────────┘
          │                           │
┌─────────▼───────────┐   ┌───────────▼───────────────────┐
│  HostKernels        │   │  OpenCL .cl kernels           │
│  csrc/kernels/host/ │   │  csrc/kernels/opencl/         │
└─────────────────────┘   │  + JIT codegen (csrc/jit/)    │
                          └───────────────────────────────┘
```

## Runtime selection

At import time, `initialize_runtime()` reads `TVARANT_BACKEND`:

| Value | Runtime | Memory |
|---|---|---|
| `sim` (default) | `CpuSimRuntime` | Host-accessible aligned buffers |
| `opencl` / `fpga` | `OpenCLRuntime` | OpenCL device buffers |

## Kernel launch model

All ops build a `LaunchParams` struct and call `runtime().launch()`:

```cpp
LaunchParams p;
p.kernel = "gemm_bias_act_kernel";
p.src0 = a.data_ptr();
p.src1 = b.data_ptr();
p.src2 = bias_ptr;
p.dst = out.data_ptr();
p.m = m; p.n = n; p.k = k;
p.act = act_id;  // 0=none, 1=relu, 2=silu
runtime().launch(p);
```

The sim runtime dispatches to C++ host functions; OpenCL enqueues NDRange kernels
with `local_work_size = 32` (Tvarant SIMT width).

## JIT layer

Pointwise fusion uses an SSA IR (`csrc/jit/Jit.h`):

- **Sim**: interpret program in a single host loop
- **OpenCL**: codegen OpenCL C → `clBuildProgram` → cache by program hash

GEMM fusion (`gemm_bias_act`) is a hand-written kernel, not JIT-generated.

## CPU fallback

Unregistered ATen ops fall back to CPU via `cpu_fallback` in `Ops.cpp`. This keeps
models running but may hurt performance — register native ops for hot paths.

## File map

| Path | Role |
|---|---|
| `csrc/Module.cpp` | pybind11 module, device API |
| `csrc/runtime/Allocator.cpp` | PyTorch allocator integration |
| `csrc/aten/Ops.cpp` | Core tensor ops |
| `csrc/aten/LlmOps.cpp` | LLM-specific ATen ops |
| `csrc/aten/CustomOps.cpp` | `tvarant.*` custom ops |
| `csrc/jit/Jit.cpp` | Pointwise IR + OpenCL codegen |
| `torch_tvarant/compiler.py` | FX graph fusion passes |
