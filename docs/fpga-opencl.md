# FPGA / OpenCL

torch_tvarant supports an OpenCL runtime path for POCL simulation and FPGA
deployment on the Tvarant stack.

## Build with OpenCL

```bash
# Linux — headers usually in /usr/include/CL
USE_OPENCL=1 pip install -e .

# Set OpenCL root if non-standard
export OPENCL_ROOT=/path/to/opencl
USE_OPENCL=1 pip install -e .
```

On Windows, point `OPENCL_ROOT` or `CUDA_PATH` at an SDK that ships `CL/cl.h`.

## Select runtime at launch

```bash
TVARANT_BACKEND=opencl python your_script.py
```

Verify:

```python
import torch_tvarant
print(torch_tvarant._C.backend())  # 'opencl'
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `TVARANT_BACKEND` | `sim` | `sim`, `opencl`, or `fpga` |
| `TVARANT_OPENCL_PLATFORM` | first platform | Substring match for platform name |
| `TVARANT_KERNEL_DIR` | embedded | Directory of `.cl` sources to load instead of embedded strings |

## Kernel sources

Shipped under `csrc/kernels/opencl/`:

```
fill.cl  copy.cl  binary.cl  relu.cl
elementwise.cl  gemm.cl  reduce.cl
```

Override at runtime:

```bash
export TVARANT_KERNEL_DIR=/path/to/custom/kernels
TVARANT_BACKEND=opencl python your_script.py
```

## OpenCL tests

```bash
TVARANT_BACKEND=opencl pytest tests/test_opencl.py -v
```

These tests are skipped unless `TVARANT_BACKEND=opencl`.

## JIT on OpenCL

Pointwise fused kernels are codegen'd and `clBuildProgram`'d at first use, then
cached. GEMM fusion uses pre-built kernels from the embedded program.

## FPGA deployment (Alveo U55C)

Target stack:

```
Python (torch_tvarant)
  → OpenCL runtime (POCL or vendor ICD)
    → libtvarant
      → xclbin on Alveo U55C
```

Deployment steps depend on your POCL/libtvarant setup. General workflow:

1. Build with `USE_OPENCL=1`
2. Configure POCL ICD / vendor OpenCL platform
3. Place xclbin where libtvarant expects it
4. Run with `TVARANT_BACKEND=opencl` (or `fpga`)

!!! note
    Full FPGA bring-up documentation will expand as hardware integration
    matures. Track progress in [GitHub Issues](https://github.com/samthakur587/Tvarant/issues).

## Windows SDK (optional)

On Windows without a system SDK, vendored headers can be placed in
`third_party/winsdk/` — see [third_party/README.md](https://github.com/samthakur587/Tvarant/blob/main/third_party/README.md).
