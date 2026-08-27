# torch.tvarant

Importing `torch_tvarant` registers the **tvarant** PrivateUse1 backend and
exposes helpers under `torch.tvarant` (same idea as `torch.cuda` / `torch.mps`).

```python
import torch
import torch_tvarant

assert torch.tvarant.is_available()
x = torch.empty(2, 3, device="tvarant")
```

Click an API below for the full signature, parameters, and examples.

## Device

| API | Summary |
|---|---|
| [`is_available`](is_available.md) | Whether the backend can run |
| [`is_initialized`](is_initialized.md) | Lazy init completed |
| [`device_count`](device_count.md) | Number of logical devices |
| [`current_device`](current_device.md) | Current device index |
| [`set_device`](set_device.md) | Select device index |
| [`synchronize`](synchronize.md) | Wait for kernels |
| [`backend`](backend.md) | `"sim"` or `"opencl"` |

## RNG

| API | Summary |
|---|---|
| [`manual_seed`](manual_seed.md) | Seed Tvarant RNG |
| [`manual_seed_all`](manual_seed_all.md) | Seed all devices |
| [`get_rng_state`](get_rng_state.md) | Read RNG state |
| [`set_rng_state`](set_rng_state.md) | Restore RNG state |

See also: [`torch_tvarant.compiler`](../compiler/index.md).
