import torch
import torch_tvarant  # noqa: F401


def test_eye_square():
    a = torch.eye(3, device="tvarant")
    assert a.device.type == "tvarant"
    assert torch.equal(a.cpu(), torch.eye(3))


def test_eye_rect():
    a = torch.eye(2, 4, device="tvarant", dtype=torch.float32)
    ref = torch.eye(2, 4, dtype=torch.float32)
    assert torch.equal(a.cpu(), ref)


def test_eye_empty():
    a = torch.eye(0, 3, device="tvarant")
    assert a.shape == (0, 3)
    assert a.device.type == "tvarant"
