# JIT compiler

The graph compiler in `torch_tvarant/compiler.py` reduces kernel launch overhead
for LLM inference by fusing common patterns into fewer device kernels.

## Usage

### FX trace + compile (recommended)

```python
import torch
import torch.nn as nn
import torch_tvarant

model = nn.Sequential(nn.Linear(768, 768), nn.ReLU()).to("tvarant")
compiled = torch_tvarant.compiler.compile(model)
y = compiled(torch.randn(4, 768, device="tvarant"))
```

### torch.compile backend

```python
compiled = torch.compile(model, backend="tvarant")
y = compiled(x)
```

The `tvarant` backend is registered automatically on import.

## Fusion passes

### 1. GEMM epilogue fusion

Patterns detected in FX graphs:

```
linear → relu/silu     →  tvarant.linear_act(x, w, bias, act, trans_b=True)
addmm  → relu/silu     →  tvarant.linear_act(...)
mm     → relu/silu     →  tvarant.linear_act(...)
mm + add → relu/silu   →  tvarant.linear_act(..., bias)
```

This replaces 2–3 kernel launches with one `gemm_bias_act_kernel`.

### 2. Pointwise chain fusion

Connected subgraphs of elementwise ops collapse into a single JIT kernel:

Supported ops: `relu`, `silu`, `add`, `mul`, `neg`, `sigmoid`, `mul.Scalar`

Example: `silu(x) * y` becomes one `tvarant.pointwise(...)` call with an SSA
program compiled at runtime.

## Direct custom ops

For manual integration or testing:

```python
# Fused linear + activation
y = torch.ops.tvarant.linear_act(x, weight, bias, "silu", trans_b=True)

# Fused pointwise (SSA program)
y = torch.ops.tvarant.pointwise(inputs, ops, a, b, input_ids, alphas, consts)
```

SSA op codes match `csrc/jit/Jit.h` (`LOAD=0`, `CONST=1`, `ADD=2`, …).

## OpenCL JIT cache

On the OpenCL backend, pointwise programs are:

1. Serialized to OpenCL C source via `codegen_opencl()`
2. Built with `clBuildProgram`
3. Cached in `OpenCLRuntime::jit_kernels_` keyed by `PointwiseProgram::cache_key()`

Repeated inference with the same fused graph reuses the compiled kernel.

## Debugging fusions

```python
from torch_tvarant.compiler import compile_fx, last_log, trace_module

gm = trace_module(model)
compiled = compile_fx(gm)
print(last_log)  # {'gemm_epilogue': 1, 'pointwise_groups': 0}
```

## Limitations

- FX tracing inlines `Linear`, `ReLU`, `SiLU`, `LayerNorm`, etc.; custom modules
  with control flow need `torch.compile` or manual op wiring
- Only `relu` and `silu` activations fuse into GEMM epilogues today
- Full transformer block fusion (attention, residuals) is on the [roadmap](roadmap.md)
