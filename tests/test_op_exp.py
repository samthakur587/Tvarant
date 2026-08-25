import torch
import torch_tvarant  # noqa: F401


def test_exp_exp2():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0], device="tvarant")
    assert torch.allclose(torch.exp(x).cpu(), torch.exp(x.cpu()), atol=1e-5)
    assert torch.allclose(torch.exp2(x).cpu(), torch.exp2(x.cpu()), atol=1e-5)
