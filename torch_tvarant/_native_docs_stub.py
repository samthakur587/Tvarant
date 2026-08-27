"""Docs-only stand-in for the native ``torch_tvarant._C`` extension.

Loaded when ``MKDOCS_BUILD=1`` and the real extension is not importable.
"""

from __future__ import annotations

import torch


def is_available() -> bool:
    return True


def device_count() -> int:
    return 1


def current_device() -> int:
    return 0


def set_device(index: int) -> None:
    _ = index


def synchronize() -> None:
    return None


def backend() -> str:
    return "sim"


def device() -> torch.device:
    return torch.device("cpu")


def manual_seed(seed: int) -> None:
    _ = seed


def get_rng_state() -> torch.Tensor:
    return torch.zeros(1, dtype=torch.uint8)


def set_rng_state(state: torch.Tensor) -> None:
    _ = state
