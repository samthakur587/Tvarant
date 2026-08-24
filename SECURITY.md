# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| main branch | Yes (development) |

## Reporting a vulnerability

If you discover a security vulnerability in torch_tvarant, please report it
responsibly.

**Do not** open a public GitHub issue for security-sensitive reports.

Instead:

1. Open a [GitHub Security Advisory](https://github.com/samthakur587/Tvarant/security/advisories/new)
   (preferred), or
2. Contact the maintainers via a private channel if you have one

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to acknowledge reports within **72 hours** and provide a fix or mitigation
plan within **14 days** for confirmed issues.

## Scope

This policy covers:

- The `torch_tvarant` Python package and C++ extension
- Build scripts (`setup.py`, CI workflows)
- Documentation site build pipeline

Out of scope: third-party dependencies (PyTorch, OpenCL drivers, FPGA toolchains).
Report those to the respective vendors.
