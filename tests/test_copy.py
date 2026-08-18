import torch
import torch_tvarant  # noqa: F401


def test_roundtrip_cpu_tvarant_cpu():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dev = src.to("tvarant")
    assert dev.device.type == "tvarant"
    back = dev.cpu()
    assert back.device.type == "cpu"
    assert torch.equal(back, src)


def test_empty_and_copy_():
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    dst = torch.empty(2, 3, device="tvarant")
    dst.copy_(src)
    assert torch.allclose(dst.cpu(), src)


def test_zeros_ones():
    z = torch.zeros(3, 4, device="tvarant")
    o = torch.ones(3, 4, device="tvarant")
    assert torch.equal(z.cpu(), torch.zeros(3, 4))
    assert torch.equal(o.cpu(), torch.ones(3, 4))


def test_to_tvarant_helper():
    x = torch.randn(5)
    y = x.tvarant()
    assert y.is_tvarant
    assert torch.allclose(y.cpu(), x)
