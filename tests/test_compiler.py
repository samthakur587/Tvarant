import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tvarant

from torch_tvarant.compiler import compile_fx, last_log, trace_module


def test_linear_act_relu_matches_eager():
    torch.manual_seed(0)
    x = torch.randn(4, 8, device="tvarant")
    w = torch.randn(16, 8, device="tvarant")
    b = torch.randn(16, device="tvarant")
    y = torch.ops.tvarant.linear_act(x, w, b, "relu", True)
    expected = F.relu(F.linear(x.cpu(), w.cpu(), b.cpu()))
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), expected, atol=1e-4)


def test_pointwise_relu_mul_add():
    a = torch.tensor([1.0, -2.0, 3.0], device="tvarant")
    b = torch.tensor([0.5, 0.5, 0.5], device="tvarant")
    # out = relu(a) * b + 1
    # SSA: v0=load0, v1=relu(v0), v2=load1, v3=v1*v2, v4=const 1, v5=v3+v4
    LOAD, CONST, ADD, MUL, RELU = 0, 1, 2, 3, 4
    y = torch.ops.tvarant.pointwise(
        [a, b],
        [LOAD, RELU, LOAD, MUL, CONST, ADD],
        [-1, 0, -1, 1, -1, 3],
        [-1, -1, -1, 2, -1, 4],
        [0, -1, 1, -1, -1, -1],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    expected = F.relu(a.cpu()) * b.cpu() + 1
    assert torch.allclose(y.cpu(), expected, atol=1e-5)


def test_compile_linear_relu_sequential():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    x = torch.randn(4, 8)
    y_cpu = model(x)
    compiled = torch_tvarant.compiler.compile(model.to("tvarant"))
    y = compiled(x.to("tvarant"))
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-4)
    assert last_log.get("gemm_epilogue", 0) >= 1


def test_compile_fx_swiglu_like():
    class SwiGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(8, 16, bias=False)
            self.w2 = nn.Linear(8, 16, bias=False)

        def forward(self, x):
            return F.silu(self.w1(x)) * self.w2(x)

    torch.manual_seed(2)
    m = SwiGLU()
    x = torch.randn(2, 8)
    y_cpu = m(x)
    compiled = compile_fx(trace_module(m.to("tvarant")))
    y = compiled(x.to("tvarant"))
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-4)
