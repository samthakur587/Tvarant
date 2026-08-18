import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tvarant  # noqa: F401


def test_linear_relu_matches_cpu():
    torch.manual_seed(0)
    cpu_net = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    x_cpu = torch.randn(4, 8)
    y_cpu = cpu_net(x_cpu)

    net = cpu_net.to("tvarant")
    x = x_cpu.to("tvarant")
    y = net(x)

    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-5)


def test_manual_linear_relu():
    torch.manual_seed(1)
    x = torch.randn(4, 8, device="tvarant")
    w = torch.randn(8, 8, device="tvarant")
    y = F.relu(x @ w)
    expected = torch.relu(x.cpu() @ w.cpu())
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), expected, atol=1e-4)


def test_two_layer_mlp():
    torch.manual_seed(2)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    x = torch.randn(2, 16)
    y_cpu = model(x)
    y = model.to("tvarant")(x.to("tvarant"))
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-4)
