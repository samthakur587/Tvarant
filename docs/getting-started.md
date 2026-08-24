# Getting started

## Requirements

| Component | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.1+ |
| C++ compiler | C++17 (GCC, Clang, or MSVC) |
| Optional | OpenCL SDK for FPGA builds |

## Install (CPU simulator)

```bash
git clone https://github.com/samthakur587/Tvarant.git
cd Tvarant
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools ninja
pip install torch pytest
pip install -e .
pytest tests/ -v
```

On **Windows**, use an x64 Native Tools prompt (or run `vcvars64.bat` first).

## Verify installation

```python
import torch
import torch_tvarant

print(torch.tvarant.is_available())   # True
print(torch.tvarant.backend())        # 'sim'
x = torch.ones(4, device="tvarant")
print(x.device)                       # tvarant:0
```

## Run a small MLP

```python
import torch
import torch.nn as nn
import torch_tvarant

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
).to("tvarant")

x = torch.randn(2, 16, device="tvarant")
y = model(x)
print(y.shape)  # torch.Size([2, 4])
```

## Compile for inference

```python
import torch_tvarant

compiled = torch_tvarant.compiler.compile(model)
y = compiled(x)
```

See [JIT Compiler](jit-compiler.md) for fusion details.

## OpenCL / FPGA

```bash
USE_OPENCL=1 pip install -e .
TVARANT_BACKEND=opencl python your_script.py
```

See [FPGA / OpenCL](fpga-opencl.md) for environment variables and kernel layout.

## Development install

```bash
pip install -e ".[dev,docs]"
pytest tests/ -v
mkdocs serve
```

See [Contributing](contributing.md) for the full contributor workflow.
