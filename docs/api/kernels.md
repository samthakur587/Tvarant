# Kernels

Kernels are the device implementations behind ATen ops. The CPU simulator calls
host functions in `csrc/kernels/host/`; the OpenCL path uses sources under
`csrc/kernels/opencl/` (also embedded in `OpenCLRuntime.cpp`).

## Host kernels (`tvarant::host`)

| Function | Role |
|---|---|
| `fill_f32` / `copy_f32` | Init and memcpy |
| `add_f32` / `mul_f32` | Binary elementwise |
| `add_scalar_f32` / `scale_f32` | Scalar elementwise |
| `relu_f32` / `silu_f32` | Activations |
| `gemm_f32` / `gemm_bias_act_f32` | GEMM (+ bias / act) |
| `bmm_f32` | Batched matmul |
| `softmax_f32` | Softmax along a dim |
| `layer_norm_f32` | Layer norm forward |
| `embedding_f32` | Embedding lookup |

Declared in `csrc/kernels/host/HostKernels.h`.

## OpenCL kernel catalog

| Kernel | File |
|---|---|
| `fill_kernel` | `fill.cl` |
| `copy_kernel` | `copy.cl` |
| `add_kernel`, `mul_kernel` | `binary.cl` |
| `relu_kernel` | `relu.cl` |
| `silu_kernel`, `scale_kernel`, `add_scalar_kernel` | `elementwise.cl` |
| `gemm_kernel`, `gemm_bias_act_kernel`, `bmm_kernel` | `gemm.cl` |
| `softmax_kernel`, `layernorm_kernel`, `embedding_kernel` | `reduce.cl` |

Pointwise JIT kernels are generated at runtime (not checked in as `.cl` files).
See [JIT Compiler](../jit-compiler.md).

## Dispatch

`CpuSimRuntime::launch` / `OpenCLRuntime::launch` select a kernel by name from
`LaunchParams.kernel`. ATen ops in `csrc/aten/` fill those params and call
`tvarant::runtime().launch(...)`.
