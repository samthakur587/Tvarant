# torch_tvarant

Out-of-tree PyTorch backend that adds `torch.device("tvarant")` for the Tvarant
RISC-V SIMT GPGPU (Xilinx Alveo U55C). Develop on a CPU simulator; the same
Python API switches to the OpenCL → POCL → libtvarant stack when the board is
available.

```python
import torch
import torch_tvarant

x = torch.randn(4, 8, device="tvarant")
w = torch.randn(8, 8, device="tvarant")
y = torch.nn.functional.relu(x @ w)
assert y.device.type == "tvarant"
```

## Install (CPU simulator)

Requires PyTorch 2.1+ and a C++17 compiler (MSVC on Windows, GCC/Clang on Linux).

```bash
pip install torch pytest ninja
pip install -e .
pytest tests/ -v
```

On Windows, run from an **x64 Native Tools** prompt (or after `vcvars64.bat`).

## JIT compiler (LLM inference)

The backend includes a graph compiler that fuses kernel launches for transformer
workloads. On the CPU simulator it runs fused host loops; on OpenCL it emits and
caches specialized kernels via `clBuildProgram`.

```python
import torch
import torch.nn as nn
import torch_tvarant

model = nn.Sequential(nn.Linear(768, 768), nn.ReLU()).to("tvarant")

# FX trace + fuse (recommended for nn.Sequential / decoder blocks)
compiled = torch_tvarant.compiler.compile(model)
y = compiled(torch.randn(4, 768, device="tvarant"))

# Or use torch.compile with the registered backend
compiled = torch.compile(model, backend="tvarant")
```

**What gets fused**

| Pattern | Fused into |
|---|---|
| `linear` / `addmm` / `mm` + bias + `relu`/`silu` | Single `gemm_bias_act` kernel |
| Pointwise chains (`relu`, `silu`, `add`, `mul`, …) | One JIT pointwise kernel |

Direct fused ops are also exposed:

```python
torch.ops.tvarant.linear_act(x, weight, bias, "silu", trans_b=True)
torch.ops.tvarant.pointwise(inputs, ops, a, b, input_ids, alphas, consts)
```

## Supported ops

Core ATen ops on `tvarant`: `empty`, `copy`, `fill`, `add`, `mul`, `relu`,
`mm`, `addmm`, `view`, `linear`.

LLM-oriented ops: `silu`, `softmax`, `matmul`, `bmm`, `layer_norm`, `embedding`,
`mul.Scalar`, `add.Scalar`.

Everything else falls back to CPU via the PrivateUse1 fallback handler.

## FPGA / OpenCL

Build with OpenCL support and select the OpenCL runtime at launch:

```bash
USE_OPENCL=1 pip install -e .
TVARANT_BACKEND=opencl python your_script.py
```

Environment variables:

| Variable | Description |
|---|---|
| `TVARANT_BACKEND` | `sim` (default) or `opencl` |
| `TVARANT_OPENCL_PLATFORM` | Substring match for OpenCL platform name |
| `TVARANT_KERNEL_DIR` | Directory of `.cl` kernel sources (defaults to embedded kernels) |

OpenCL kernel sources live in `csrc/kernels/opencl/`.
