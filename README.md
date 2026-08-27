# torch_tvarant

[![CI](https://github.com/samthakur587/Tvarant/actions/workflows/ci.yml/badge.svg)](https://github.com/samthakur587/Tvarant/actions/workflows/ci.yml)
[![Docs](https://github.com/samthakur587/Tvarant/actions/workflows/docs.yml/badge.svg)](https://github.com/samthakur587/Tvarant/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

**Open-source PyTorch backend for the Tvarant RISC-V SIMT GPGPU** (Xilinx Alveo U55C).

Develop on a CPU simulator, deploy through OpenCL → POCL → libtvarant when hardware
is available. Includes a JIT graph compiler for LLM inference fusion.

📖 **[Documentation](https://samthakur587.github.io/Tvarant/)** ·
🐛 [Issues](https://github.com/samthakur587/Tvarant/issues) ·
🗺️ [Roadmap](docs/roadmap.md) ·
🤝 [Contributing](CONTRIBUTING.md)

## Quick start

```python
import torch
import torch_tvarant

x = torch.randn(4, 8, device="tvarant")
w = torch.randn(8, 8, device="tvarant")
y = torch.nn.functional.relu(x @ w)
assert y.device.type == "tvarant"
```

```bash
pip install torch pytest ninja
pip install -e .
pytest tests/ -v
```

See [Installation](docs/install.md) and [Getting Started](docs/getting-started.md).
API: [docs/api](docs/api/index.md).

## Features

| Feature | Description |
|---|---|
| **Device backend** | `torch.device("tvarant")` via PrivateUse1 |
| **CPU simulator** | Host-accessible memory for fast iteration |
| **OpenCL runtime** | Same API on POCL / FPGA path |
| **LLM ops** | `silu`, `softmax`, `bmm`, `layer_norm`, `embedding`, … |
| **Fused GEMM** | Single kernel for matmul + bias + relu/silu |
| **JIT compiler** | FX fusion + OpenCL kernel codegen for pointwise chains |
| **torch.compile** | Registered `backend="tvarant"` |

## JIT compiler

```python
import torch
import torch.nn as nn
import torch_tvarant

model = nn.Sequential(nn.Linear(768, 768), nn.ReLU()).to("tvarant")
compiled = torch_tvarant.compiler.compile(model)
y = compiled(torch.randn(4, 768, device="tvarant"))
```

Fuses `linear + relu/silu` into one kernel and collapses pointwise chains.
Details: [JIT Compiler docs](docs/jit-compiler.md).

## FPGA / OpenCL

```bash
USE_OPENCL=1 pip install -e .
TVARANT_BACKEND=opencl python your_script.py
```

See [FPGA / OpenCL docs](docs/fpga-opencl.md).

## Project structure

```
csrc/aten/          ATen op registrations
csrc/jit/           Pointwise JIT IR + OpenCL codegen
csrc/kernels/       Host + OpenCL kernels
csrc/runtime/       Sim and OpenCL device runtime
torch_tvarant/      Python package + graph compiler
tests/              Pytest suite
docs/               MkDocs documentation
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Open a pull request

Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Roadmap

- Fused attention kernel ([#1](https://github.com/samthakur587/Tvarant/issues/1))
- KV-cache + decode GEMM ([#2](https://github.com/samthakur587/Tvarant/issues/2))
- RMSNorm, RoPE, SwiGLU ([#3](https://github.com/samthakur587/Tvarant/issues/3))
- Shape-specialized GEMM JIT ([#4](https://github.com/samthakur587/Tvarant/issues/4))

Full roadmap: [docs/roadmap.md](docs/roadmap.md)

## License

Licensed under the [Apache License 2.0](LICENSE).

## Security

Report vulnerabilities via [GitHub Security Advisories](https://github.com/samthakur587/Tvarant/security/advisories/new).
See [SECURITY.md](SECURITY.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
