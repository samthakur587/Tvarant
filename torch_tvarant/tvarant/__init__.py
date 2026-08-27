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
    """Return whether the Tvarant backend extension is usable.

    After a successful ``import torch_tvarant``, this is normally ``True``.

    Returns:
        bool: ``True`` if kernels can run on ``device="tvarant"``.

    Examples:
        >>> import torch_tvarant
        >>> torch.tvarant.is_available()
        True
    """
    return bool(_C.is_available())


def is_initialized() -> bool:
    """Return whether lazy device initialization has completed.

    Returns:
        bool: ``True`` after the first device API call that triggers init.
    """
    return _initialized


def device_count() -> int:
    """Return the number of logical Tvarant devices.

    Returns:
        int: Device count (currently ``1`` for both sim and OpenCL backends).

    Examples:
        >>> torch.tvarant.device_count()
        1
    """
    return int(_C.device_count())


def current_device() -> int:
    """Return the current device index.

    Returns:
        int: Active device ordinal (``0`` today).
    """
    return int(_C.current_device())


def set_device(device: int) -> None:
    """Set the current device index.

    Args:
        device: Device ordinal. Must be in ``[0, device_count())``.

    Examples:
        >>> torch.tvarant.set_device(0)
    """
    _C.set_device(int(device))


def synchronize(device: int | None = None) -> None:
    """Wait for outstanding work on the Tvarant runtime.

    Args:
        device: Reserved for API compatibility with other PyTorch device
            modules. Ignored; all work shares one runtime queue today.
    """
    _ = device
    _C.synchronize()


def backend() -> str:
    """Return the active runtime name.

    Returns:
        str: ``"sim"`` for the CPU simulator, or ``"opencl"`` when built and
        selected via ``TVARANT_BACKEND=opencl``.

    Examples:
        >>> torch.tvarant.backend()
        'sim'
    """
    return str(_C.backend())


def _is_in_bad_fork() -> bool:
    return False


def manual_seed(seed: int) -> None:
    """Seed the Tvarant generator only.

    Does **not** call ``torch.manual_seed``, which would re-enter
    ``torch.tvarant.manual_seed_all`` and recurse.

    Args:
        seed: Integer seed for the PrivateUse1 default generator.
    """
    _C.manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    """Seed all Tvarant devices.

    Args:
        seed: Integer seed applied to every logical device (currently one).
    """
    manual_seed(seed)


def get_rng_state(device: Union[int, str, torch.device] = "tvarant") -> torch.Tensor:
    """Return the RNG state tensor for ``device``.

    Args:
        device: Device specifier. Accepted for API compatibility; only the
            single Tvarant generator is used today.

    Returns:
        torch.Tensor: Opaque byte state suitable for ``set_rng_state``.
    """
    _ = device
    return _C.get_rng_state()


def set_rng_state(
    new_state: torch.Tensor, device: Union[int, str, torch.device] = "tvarant"
) -> None:
    """Restore RNG state previously returned by ``get_rng_state``.

    Args:
        new_state: State tensor from ``get_rng_state``.
        device: Device specifier (compatibility; ignored beyond validation).
    """
    _ = device
    _C.set_rng_state(new_state)
