import os

import pytest
import torch
import torch_tvarant  # noqa: F401


def test_backend_name():
    assert torch.utils.backend_registration._privateuse1_backend_name == "tvarant"


def test_is_available():
    assert torch.tvarant.is_available()
    assert torch.tvarant.device_count() == 1
    assert torch.tvarant.current_device() == 0


def test_device_object():
    d = torch.device("tvarant")
    assert d.type == "tvarant"
    d0 = torch.device("tvarant:0")
    assert d0.index == 0


def test_tensor_helpers():
    x = torch.empty(2, 3, device="tvarant")
    assert x.is_tvarant
    assert x.device.type == "tvarant"
    y = torch.empty(2, 3)
    assert not y.is_tvarant
    z = y.tvarant()
    assert z.is_tvarant


@pytest.mark.skipif(
    os.environ.get("TVARANT_BACKEND", "sim").lower() not in {"", "sim"},
    reason="default backend check is sim-only",
)
def test_backend_default_is_sim():
    assert torch_tvarant._C.backend() == "sim"
