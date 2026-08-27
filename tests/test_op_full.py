import torch
import torch_tvarant  # noqa: F401


def test_full():
    f = torch.full((2, 3), 2.5, device="tvarant")
    assert f.device.type == "tvarant"
    assert torch.allclose(f.cpu(), torch.full((2, 3), 2.5))
    z = torch.zeros(2, 3, device="tvarant")
    assert torch.allclose(torch.full_like(z, 7.0).cpu(), torch.full((2, 3), 7.0))
