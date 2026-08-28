import torch
import torch_tvarant  # noqa: F401


def test_arange_end():
    a = torch.arange(5, device="tvarant")
    assert a.device.type == "tvarant"
    assert torch.equal(a.cpu(), torch.arange(5))


def test_arange_start_step():
    a = torch.arange(2, 10, 2, device="tvarant")
    assert torch.equal(a.cpu(), torch.arange(2, 10, 2))


def test_arange_float():
    a = torch.arange(0.0, 1.0, 0.25, device="tvarant", dtype=torch.float32)
    ref = torch.arange(0.0, 1.0, 0.25, dtype=torch.float32)
    assert torch.allclose(a.cpu(), ref, atol=1e-6)
