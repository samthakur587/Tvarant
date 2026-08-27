import torch
import torch_tvarant  # noqa: F401


def test_clone():
    x = torch.randn(3, 4, device="tvarant")
    y = x.clone()
    assert y.device.type == "tvarant"
    assert y.data_ptr() != x.data_ptr()
    assert torch.allclose(y.cpu(), x.cpu())
