# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-24

### Added
- GitHub Actions CI, release, docs, and stale-issue workflows
- Apache 2.0 license, contribution guidelines, code of conduct, and security policy
- MkDocs documentation site under `docs/`
- Issue and pull request templates, Dependabot configuration
- Out-of-tree PyTorch backend registering `torch.device("tvarant")`
- CPU simulator runtime with host-accessible device memory
- Optional OpenCL runtime (`USE_OPENCL=1`, `TVARANT_BACKEND=opencl`)
- Core ATen ops: `empty`, `copy`, `fill`, `add`, `mul`, `relu`, `mm`, `addmm`, `linear`, `view`
- LLM ops: `silu`, `softmax`, `matmul`, `bmm`, `layer_norm`, `embedding`
- Fused `gemm_bias_act` kernel (GEMM + bias + relu/silu)
- JIT pointwise compiler with OpenCL kernel codegen and cache
- `torch_tvarant.compiler` FX fusion passes and `torch.compile` backend
- Custom ops: `tvarant.linear_act`, `tvarant.pointwise`
- Pytest suite for ops, MLP, compiler, and LLM building blocks

[Unreleased]: https://github.com/samthakur587/Tvarant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/samthakur587/Tvarant/releases/tag/v0.1.0
