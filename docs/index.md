# torch_tvarant documentation

**torch_tvarant** is an out-of-tree PyTorch device backend for the
[Tvarant](https://github.com/samthakur587/Tvarant) RISC-V SIMT GPGPU — targeting
the Xilinx Alveo U55C FPGA accelerator.

It lets you write standard PyTorch code and run it on `torch.device("tvarant")`:

```python
import torch
import torch_tvarant

x = torch.randn(4, 768, device="tvarant")
w = torch.randn(768, 768, device="tvarant")
y = torch.nn.functional.relu(x @ w)
```

## Key features

- **CPU simulator** — develop and test without hardware
- **OpenCL runtime** — same Python API on POCL / FPGA path
- **LLM ops** — `silu`, `softmax`, `bmm`, `layer_norm`, `embedding`, and more
- **JIT compiler** — fuse GEMM epilogues and pointwise chains for inference
- **Extensible** — add kernels in C++/OpenCL and register ATen ops

## Quick links

| Resource | Link |
|---|---|
| Installation | [install.md](install.md) |
| Getting started | [getting-started.md](getting-started.md) |
| API reference | [api/index.md](api/index.md) |
| Architecture | [architecture.md](architecture.md) |
| JIT compiler | [jit-compiler.md](jit-compiler.md) |
| Contributing | [contributing.md](contributing.md) |
| GitHub | [samthakur587/Tvarant](https://github.com/samthakur587/Tvarant) |
| Changelog | [CHANGELOG.md](https://github.com/samthakur587/Tvarant/blob/main/CHANGELOG.md) |

## License

Apache License 2.0 — see [LICENSE](https://github.com/samthakur587/Tvarant/blob/main/LICENSE).
