"""Pytest bootstrap: import the installed torch_tvarant backend."""

from __future__ import annotations

import importlib.util
import sys

import torch

# Prefer the installed package (with _C.so). If the repo checkout is on
# sys.path ahead of site-packages, imports resolve to source without the
# extension and fail mysteriously.
_spec = importlib.util.find_spec("torch_tvarant._C")
if _spec is None or _spec.origin is None:
    raise ImportError(
        "torch_tvarant._C extension not found. Install first with:\n"
        "  pip install . --no-build-isolation\n"
        "Then run pytest from outside the repo (or use an editable install).\n"
        f"sys.path[0]={sys.path[0]!r}"
    )

import torch_tvarant  # noqa: E402, F401


def pytest_configure():
    assert torch.tvarant.is_available()
