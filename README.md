# torch_tvarant

Out-of-tree PyTorch backend that adds `torch.device("tvarant")` for the Tvarant
RISC-V SIMT GPGPU (Xilinx Alveo U55C). Develop on a CPU simulator; the same
Python API switches to the existing OpenCL → POCL → libtvarant stack when the
board is available.

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

## FPGA

See [docs/tvarant-torch.md](docs/tvarant-torch.md) for `TVARANT_BACKEND=opencl`,
POCL ICD env, xclbin path, and how to vendor Darknet `.cl` kernels.
