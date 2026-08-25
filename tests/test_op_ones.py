import torch
import torch_tvarant  # noqa: F401


def test_ones():
    o = torch.ones(2, 3, device="tvarant")
    assert o.device.type == "tvarant"
    assert torch.allclose(o.cpu(), torch.ones(2, 3))
    z = torch.zeros(2, 3, device="tvarant")
    assert torch.allclose(torch.ones_like(z).cpu(), torch.ones(2, 3))
