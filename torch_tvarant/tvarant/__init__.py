"""Python device module registered as ``torch.tvarant``.

Functions here mirror the usual PyTorch device-module surface
(``is_available``, ``synchronize``, RNG helpers, …) and forward to the
native extension.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Union

import torch

try:
    _C = importlib.import_module("torch_tvarant._C")
except ImportError:
    if os.environ.get("MKDOCS_BUILD") == "1":
        from torch_tvarant import _native_docs_stub as _C

        sys.modules["torch_tvarant._C"] = _C
    else:
        raise

_initialized = False


def _lazy_init() -> None:
    global _initialized
    if _initialized:
        return
    _C.is_available()
    _initialized = True


def is_available() -> bool:
    """Return whether the Tvarant backend extension is usable."""
    return bool(_C.is_available())


def is_initialized() -> bool:
    """Return whether lazy device initialization has completed."""
    return _initialized


def device_count() -> int:
    """Return the number of logical Tvarant devices."""
    return int(_C.device_count())


def current_device() -> int:
    """Return the current device index."""
    return int(_C.current_device())


def set_device(device: int) -> None:
    """Set the current device index."""
    _C.set_device(int(device))


def synchronize(device: int | None = None) -> None:
    """Wait for outstanding work on the runtime (``device`` reserved)."""
    _ = device
    _C.synchronize()


def backend() -> str:
    """Return the active runtime name (``sim`` or ``opencl``)."""
    return str(_C.backend())


def _is_in_bad_fork() -> bool:
    return False


def manual_seed(seed: int) -> None:
    """Seed the Tvarant generator only (does not call ``torch.manual_seed``)."""
    _C.manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    """Seed all Tvarant devices (currently one device)."""
    manual_seed(seed)


def get_rng_state(device: Union[int, str, torch.device] = "tvarant") -> torch.Tensor:
    """Return the RNG state tensor for ``device``."""
    _ = device
    return _C.get_rng_state()


def set_rng_state(
    new_state: torch.Tensor, device: Union[int, str, torch.device] = "tvarant"
) -> None:
    """Restore RNG state for ``device``."""
    _ = device
    _C.set_rng_state(new_state)
