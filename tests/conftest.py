import torch
import torch_tvarant  # noqa: F401


def pytest_configure():
    assert torch.tvarant.is_available()
