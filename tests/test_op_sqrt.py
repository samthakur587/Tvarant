import torch
import torch_tvarant  # noqa: F401


def test_sqrt_rsqrt():
    pos = torch.tensor([0.25, 1.0, 4.0], device="tvarant")
    assert torch.allclose(torch.sqrt(pos).cpu(), torch.sqrt(pos.cpu()), atol=1e-5)
    assert torch.allclose(torch.rsqrt(pos).cpu(), torch.rsqrt(pos.cpu()), atol=1e-5)
