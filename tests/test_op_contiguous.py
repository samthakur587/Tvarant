import torch
import torch_tvarant  # noqa: F401


def test_contiguous():
    x = torch.randn(3, 4, device="tvarant")
    v = x.t()
    c = v.contiguous()
    assert c.is_contiguous()
    assert torch.allclose(c.cpu(), v.cpu())
