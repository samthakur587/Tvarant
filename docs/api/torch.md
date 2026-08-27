# torch.tvarant

Importing `torch_tvarant` registers the **tvarant** PrivateUse1 backend and exposes
helpers under `torch.tvarant` (same surface as other PyTorch device modules).

```python
import torch
import torch_tvarant

torch.tvarant.is_available()
torch.tvarant.device_count()
x = torch.empty(2, 3, device="tvarant")
```

## Device helpers

| Function | Description |
|---|---|
| `is_available()` | Backend extension loaded and usable |
| `is_initialized()` | Lazy init has run |
| `device_count()` | Number of logical devices (currently 1) |
| `current_device()` | Current device index |
| `set_device(index)` | Set current device index |
| `synchronize(device=None)` | Wait for outstanding kernels |
| `backend()` | Runtime name: `'sim'` or `'opencl'` |
| `manual_seed(seed)` | Seed the Tvarant RNG |
| `manual_seed_all(seed)` | Alias of `manual_seed` |
| `get_rng_state(device=...)` | RNG state tensor |
| `set_rng_state(state, device=...)` | Restore RNG state |

These map to bindings in `csrc/Module.cpp`.

## Package modules

Auto-generated from Python docstrings (requires `MKDOCS_BUILD=1` when the
native extension is not built).

::: torch_tvarant
    options:
      members: false
      show_root_heading: true
      show_source: false

::: torch_tvarant.tvarant
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      docstring_style: google
      filters:
        - "!^_"

::: torch_tvarant.compiler
    options:
      show_root_heading: true
      show_source: false
      members:
        - compile
        - compile_fx
        - register
        - trace_module
        - fuse_gemm_epilogue
        - fuse_pointwise
        - TvarantTracer
      members_order: source
