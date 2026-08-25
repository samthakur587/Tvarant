import torch
import torch_tvarant  # noqa: F401


def test_abs():
    x = torch.tensor([-4.0, -1.0, 0.25, 4.0], device="tvarant")
    assert torch.allclose(torch.abs(x).cpu(), torch.abs(x.cpu()))
