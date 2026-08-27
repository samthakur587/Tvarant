# Supported ops

All listed ops run on `torch.device("tvarant")` unless they fall back to CPU.

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

## Adding a new op

See [C++ API](cpp.md) and [Contributing](../contributing.md).
