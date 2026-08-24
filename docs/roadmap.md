# Roadmap

Track active work on [GitHub Issues](https://github.com/samthakur587/Tvarant/issues).

## Completed (v0.1.0)

- [x] CPU simulator runtime
- [x] OpenCL runtime with embedded kernels
- [x] Core ATen ops and LLM ops (silu, softmax, bmm, layer_norm, embedding)
- [x] Fused GEMM + bias + activation kernel
- [x] JIT pointwise compiler with OpenCL codegen
- [x] `torch_tvarant.compiler` FX fusion passes
- [x] CI, docs, and release infrastructure

## In progress / planned

| Priority | Item | Issue |
|---|---|---|
| High | Fused attention (QK^T + softmax + PV) | [#1](https://github.com/samthakur587/Tvarant/issues/1) |
| High | KV-cache + decode GEMM (M=1) | [#2](https://github.com/samthakur587/Tvarant/issues/2) |
| High | RMSNorm, RoPE, SwiGLU ops | [#3](https://github.com/samthakur587/Tvarant/issues/3) |
| Medium | Shape-specialized GEMM JIT | [#4](https://github.com/samthakur587/Tvarant/issues/4) |
| Medium | Full transformer block fusion | [#7](https://github.com/samthakur587/Tvarant/issues/7) |
| Long-term | Lower to Tvarant RISC-V ISA | [#5](https://github.com/samthakur587/Tvarant/issues/5) |
| Long-term | int8 / FP16 weight quantization | [#6](https://github.com/samthakur587/Tvarant/issues/6) |

## How to pick up work

1. Comment on an issue to claim it
2. Fork and branch from `main`
3. Follow [Contributing](contributing.md)
4. Open a PR linking the issue

## Versioning

We use [Semantic Versioning](https://semver.org/):

- **Patch** — bug fixes, kernel correctness
- **Minor** — new ops, compiler fusions, backward-compatible features
- **Major** — breaking API or ABI changes

See [CHANGELOG.md](https://github.com/samthakur587/Tvarant/blob/main/CHANGELOG.md).
