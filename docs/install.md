# Installation

## Requirements

| Component | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.1+ |
| C++ compiler | C++17 (GCC, Clang, or MSVC) |
| Optional | OpenCL SDK for FPGA builds |

## Install from source (CPU simulator)

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

## Install from PyPI

When a release is published:

```bash
pip install torch-tvarant
```

Use a matching CPU or CUDA PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) first.

## OpenCL / FPGA build

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

See [Contributing](contributing.md) for the contributor workflow.

## Next steps

Continue with [Getting Started](getting-started.md).
