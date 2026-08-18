"""Out-of-tree PyTorch backend that registers torch.device('tvarant')."""

from __future__ import annotations

import torch

from . import _C
from . import tvarant

torch.utils.rename_privateuse1_backend("tvarant")
torch._register_device_module("tvarant", tvarant)
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
)

__all__ = ["tvarant", "_C"]
__version__ = "0.1.0"


def _autoload() -> None:
    """Entry point for TORCH_DEVICE_BACKEND_AUTOLOAD."""
