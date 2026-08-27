import torch
import torch_tvarant  # noqa: F401


def test_zeros():
    z = torch.zeros(2, 3, device="tvarant")
    assert z.device.type == "tvarant"
    assert torch.allclose(z.cpu(), torch.zeros(2, 3))
    o = torch.ones(2, 3, device="cpu").to("tvarant")
    assert torch.allclose(torch.zeros_like(o).cpu(), torch.zeros(2, 3))
