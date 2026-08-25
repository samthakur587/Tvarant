import torch
import torch_tvarant  # noqa: F401


def test_sub():
    a = torch.tensor([5.0, 7.0, 9.0], device="tvarant")
    b = torch.tensor([1.0, 2.0, 3.0], device="tvarant")
    assert torch.allclose((a - b).cpu(), torch.tensor([4.0, 5.0, 6.0]))
    assert torch.allclose((a - 1.5).cpu(), a.cpu() - 1.5, atol=1e-5)
