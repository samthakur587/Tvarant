import torch
import torch_tvarant  # noqa: F401


def test_linspace_basic():
    a = torch.linspace(0, 1, 5, device="tvarant")
    assert a.device.type == "tvarant"
    ref = torch.linspace(0, 1, 5)
    assert torch.allclose(a.cpu(), ref, atol=1e-6)


def test_linspace_single_step():
    a = torch.linspace(3.0, 7.0, 1, device="tvarant", dtype=torch.float32)
    ref = torch.linspace(3.0, 7.0, 1, dtype=torch.float32)
    assert torch.allclose(a.cpu(), ref, atol=1e-6)


def test_linspace_empty():
    a = torch.linspace(0, 1, 0, device="tvarant")
    assert a.numel() == 0
    assert a.device.type == "tvarant"
