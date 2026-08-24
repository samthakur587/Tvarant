"""Python device module registered as torch.tvarant."""

from __future__ import annotations

import importlib
from typing import Union

import torch

# Import the extension module directly so this package can load while
# torch_tvarant/__init__.py is still initializing (avoids circular import).
_C = importlib.import_module("torch_tvarant._C")

_initialized = False


def _lazy_init() -> None:
    global _initialized
    if _initialized:
        return
    _C.is_available()
    _initialized = True


def is_available() -> bool:
    return bool(_C.is_available())


def is_initialized() -> bool:
    return _initialized


def device_count() -> int:
    return int(_C.device_count())


def current_device() -> int:
    return int(_C.current_device())


def set_device(device: int) -> None:
    _C.set_device(int(device))


def synchronize(device: int | None = None) -> None:
    _ = device
    _C.synchronize()


def backend() -> str:
    return str(_C.backend())


def _is_in_bad_fork() -> bool:
    return False


def manual_seed(seed: int) -> None:
    # Seed only the Tvarant generator. Do not call torch.manual_seed — that
    # re-enters torch.tvarant.manual_seed_all and recurses.
    _C.manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    manual_seed(seed)


def get_rng_state(device: Union[int, str, torch.device] = "tvarant") -> torch.Tensor:
    _ = device
    return _C.get_rng_state()


def set_rng_state(
    new_state: torch.Tensor, device: Union[int, str, torch.device] = "tvarant"
) -> None:
    _ = device
    _C.set_rng_state(new_state)
