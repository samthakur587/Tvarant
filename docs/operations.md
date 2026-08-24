# Operations

All ops run on `torch.device("tvarant")` unless they fall back to CPU.

## Core ATen ops

| Op | Notes |
|---|---|
| `empty`, `empty_strided` | Strided layout only |
| `as_strided`, `view`, `resize_` | View/resize support |
| `_copy_from` | H2D, D2H, D2D via runtime |
| `fill_` | Scalar fill |
| `add`, `mul` | Broadcast elementwise |
| `relu` | Elementwise ReLU |
| `mm`, `addmm` | Matrix multiply; uses fused GEMM path |
| `linear` | `nn.Linear` backend; transposed weight |

## LLM ops

| Op | Notes |
|---|---|
| `silu` | SiLU / Swish activation |
| `softmax` | Last-dim or arbitrary dim |
| `matmul` | 2D GEMM or batched BMM |
| `bmm` | Batch matrix multiply |
| `native_layer_norm` | Forward layer norm + mean/rstd |
| `embedding` | Token lookup (int32 indices) |
| `mul.Scalar`, `add.Scalar` | Scalar elementwise |

## Custom ops (`torch.ops.tvarant.*`)

| Op | Signature | Description |
|---|---|---|
| `linear_act` | `(x, weight, bias?, act, trans_b=False)` | Fused GEMM + bias + relu/silu |
| `pointwise` | `(inputs, ops, a, b, input_ids, alphas, consts)` | JIT fused elementwise program |

Activation strings for `linear_act`: `"none"`, `"relu"`, `"silu"`.

## CPU fallback

Any ATen op not registered for `PrivateUse1` executes on CPU via the fallback
handler. Tensors are copied to/from device memory as needed.

Check for unexpected CPU fallback during profiling:

```python
import torch
torch.autograd.set_detect_anomaly(False)
# Watch for slow ops or CPU device in stack traces
```

## Adding a new op

1. Implement host kernel in `csrc/kernels/host/HostKernels.{h,cpp}`
2. Add OpenCL kernel in `csrc/kernels/opencl/` and embed in `OpenCLRuntime.cpp`
3. Wire dispatch in `CpuSimRuntime::launch` and `OpenCLRuntime::launch`
4. Register ATen impl in `Ops.cpp` or `LlmOps.cpp`
5. Add pytest in `tests/`

See [Contributing](contributing.md) for the full workflow.

## OpenCL kernel catalog

| Kernel | File |
|---|---|
| `fill_kernel` | `fill.cl` |
| `copy_kernel` | `copy.cl` |
| `add_kernel`, `mul_kernel` | `binary.cl` |
| `relu_kernel` | `relu.cl` |
| `silu_kernel`, `scale_kernel` | `elementwise.cl` |
| `gemm_kernel`, `gemm_bias_act_kernel`, `bmm_kernel` | `gemm.cl` |
| `softmax_kernel`, `layernorm_kernel`, `embedding_kernel` | `reduce.cl` |

Pointwise JIT kernels are generated at runtime (not in `.cl` files).
