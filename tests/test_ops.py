import torch
import torch.nn.functional as F
import torch_tvarant  # noqa: F401


def test_fill():
    x = torch.empty(4, device="tvarant")
    x.fill_(2.5)
    assert torch.allclose(x.cpu(), torch.full((4,), 2.5))


def test_add():
    a = torch.tensor([1.0, 2.0, 3.0], device="tvarant")
    b = torch.tensor([4.0, 5.0, 6.0], device="tvarant")
    c = a + b
    assert c.device.type == "tvarant"
    assert torch.allclose(c.cpu(), torch.tensor([5.0, 7.0, 9.0]))


def test_add_broadcast():
    a = torch.ones(2, 3, device="tvarant")
    b = torch.tensor([1.0, 2.0, 3.0], device="tvarant")
    c = a + b
    assert torch.allclose(c.cpu(), torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]))


def test_mul():
    a = torch.tensor([2.0, 3.0], device="tvarant")
    b = torch.tensor([4.0, 5.0], device="tvarant")
    assert torch.allclose((a * b).cpu(), torch.tensor([8.0, 15.0]))


def test_relu():
    x = torch.tensor([-1.0, 0.0, 2.5], device="tvarant")
    y = F.relu(x)
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), torch.tensor([0.0, 0.0, 2.5]))


def test_mm():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="tvarant")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="tvarant")
    c = a @ b
    assert c.device.type == "tvarant"
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(c.cpu(), expected)


def test_addmm_like_linear():
    x = torch.tensor([[1.0, 2.0]], device="tvarant")
    w = torch.tensor([[0.5, 0.0], [0.0, 0.25]], device="tvarant")
    b = torch.tensor([1.0, -1.0], device="tvarant")
    y = torch.addmm(b, x, w.t())
    expected = torch.addmm(b.cpu(), x.cpu(), w.cpu().t())
    assert torch.allclose(y.cpu(), expected)


def test_item():
    x = torch.tensor(3.0, device="tvarant")
    assert x.item() == 3.0
