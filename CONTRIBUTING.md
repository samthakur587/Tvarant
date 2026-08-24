# Contributing to torch_tvarant

Thank you for your interest in contributing to **torch_tvarant** — the open-source
PyTorch backend for the Tvarant RISC-V SIMT GPGPU.

## Ways to contribute

- Report bugs and request features via [GitHub Issues](https://github.com/samthakur587/Tvarant/issues)
- Improve documentation in `docs/` and `README.md`
- Add or optimize kernels in `csrc/kernels/`
- Extend the JIT compiler in `torch_tvarant/compiler.py` and `csrc/jit/`
- Add tests in `tests/`
- Review pull requests

## Development setup

### Prerequisites

- Python 3.9+
- PyTorch 2.1+
- C++17 compiler (GCC/Clang on Linux, MSVC on Windows)
- Optional: OpenCL SDK for `USE_OPENCL=1` builds

### Clone and install

```bash
git clone https://github.com/samthakur587/Tvarant.git
cd Tvarant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel setuptools ninja
pip install torch pytest
pip install -e ".[dev]"
pytest tests/ -v
```

### OpenCL / FPGA backend

```bash
USE_OPENCL=1 pip install -e .
TVARANT_BACKEND=opencl pytest tests/test_opencl.py -v
```

### Build docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open http://127.0.0.1:8000 to preview the documentation site.

## Project layout

| Path | Purpose |
|---|---|
| `csrc/aten/` | PyTorch ATen op registrations |
| `csrc/jit/` | Pointwise JIT IR and OpenCL codegen |
| `csrc/kernels/host/` | CPU simulator kernels |
| `csrc/kernels/opencl/` | OpenCL kernel sources |
| `csrc/runtime/` | Device runtime (sim + OpenCL) |
| `torch_tvarant/` | Python package and graph compiler |
| `tests/` | Pytest suite |
| `docs/` | MkDocs documentation |

## Coding guidelines

- Match existing C++ style in `csrc/` (C++17, minimal scope changes)
- Register new ATen ops under the `PrivateUse1` dispatch key
- Add pytest coverage for every new op or fusion pattern
- Keep host and OpenCL kernel paths in sync when adding kernels
- Prefer fused kernels over many small launches for LLM workloads

## Pull request process

1. Fork the repository and create a feature branch from `main`
2. Make focused changes with clear commit messages
3. Run the full test suite: `pytest tests/ -v`
4. Update `CHANGELOG.md` under **Unreleased** for user-visible changes
5. Update docs if you change APIs, env vars, or build steps
6. Open a PR and link related issues (e.g. `Fixes #12`)

CI must pass before merge. Maintainers will review for correctness,
performance impact, and API consistency.

## Commit messages

Use clear, imperative subject lines:

```
Add fused attention kernel for batched MHA

Implement QK^T + softmax + PV in one launch on sim and OpenCL backends.
```

## Release process (maintainers)

1. Update version in `pyproject.toml` and `torch_tvarant/__init__.py`
2. Move **Unreleased** entries in `CHANGELOG.md` to a new version section
3. Tag: `git tag v0.2.0 && git push origin v0.2.0`
4. GitHub Actions (`.github/workflows/release.yml`) runs tests, builds an sdist,
   creates a GitHub Release, and publishes to PyPI via **Trusted Publishing**
   (OIDC). Configure a PyPI publisher for project `torch-tvarant`, workflow
   `release.yml`, repository `samthakur587/Tvarant` — no API token required.

## Community

Please read our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful and
constructive in all project spaces.

## Questions?

Open a [Discussion](https://github.com/samthakur587/Tvarant/discussions) or
an issue with the `question` label.
