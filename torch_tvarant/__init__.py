"""Out-of-tree PyTorch backend that registers ``torch.device("tvarant")``.

Importing this package:

* loads the native ``_C`` extension (or a docs stub when ``MKDOCS_BUILD=1``)
* renames PrivateUse1 to ``tvarant``
* exposes device helpers under ``torch.tvarant``
"""

from __future__ import annotations

import os
import sys

import torch

try:
    from . import _C
except ImportError:
    if os.environ.get("MKDOCS_BUILD") == "1":
        from . import _native_docs_stub as _C

        sys.modules[__name__ + "._C"] = _C
    else:
        raise

from . import compiler, tvarant

torch.utils.rename_privateuse1_backend("tvarant")
torch._register_device_module("tvarant", tvarant)
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
)

__all__ = ["tvarant", "_C", "compiler"]
__version__ = "0.1.0"


def _autoload() -> None:
    """Entry point for ``TORCH_DEVICE_BACKEND_AUTOLOAD``."""
