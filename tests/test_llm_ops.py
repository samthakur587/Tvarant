import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tvarant  # noqa: F401


def test_silu_matches_cpu():
    x = torch.randn(4, 8)
    y = F.silu(x.to("tvarant"))
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), F.silu(x), atol=1e-5)


def test_softmax_matches_cpu():
    x = torch.randn(2, 4, 8)
    y = F.softmax(x.to("tvarant"), dim=-1)
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), F.softmax(x, dim=-1), atol=1e-5)


def test_layer_norm_matches_cpu():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    ln = nn.LayerNorm(8)
    y_cpu = ln(x)
    y = ln.to("tvarant")(x.to("tvarant"))
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-5)


def test_bmm_matches_cpu():
    a = torch.randn(3, 4, 5)
    b = torch.randn(3, 5, 6)
    y = torch.bmm(a.to("tvarant"), b.to("tvarant"))
    assert torch.allclose(y.cpu(), torch.bmm(a, b), atol=1e-4)


def test_matmul_batched():
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 8, 5)
    y = q.to("tvarant") @ k.to("tvarant")
    assert y.shape == (2, 3, 4, 5)
    assert torch.allclose(y.cpu(), q @ k, atol=1e-4)


def test_mul_scalar():
    x = torch.randn(4, 8, device="tvarant")
    y = x * 0.125
    assert torch.allclose(y.cpu(), x.cpu() * 0.125, atol=1e-6)


def test_embedding_matches_cpu():
    torch.manual_seed(0)
    emb = nn.Embedding(16, 8)
    idx = torch.tensor([[1, 3, 0], [2, 2, 7]])
    y_cpu = emb(idx)
    y = emb.to("tvarant")(idx)
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-6)


def test_tiny_attention_block():
    torch.manual_seed(0)
    b, t, d = 2, 4, 8
    x = torch.randn(b, t, d)
    wq = torch.randn(d, d)
    wk = torch.randn(d, d)
    wv = torch.randn(d, d)
    scale = d**-0.5

    def attn(x_, wq_, wk_, wv_):
        q = x_ @ wq_
        k = x_ @ wk_
        v = x_ @ wv_
        scores = (q @ k.transpose(-2, -1)) * scale
        return F.softmax(scores, dim=-1) @ v

    y_cpu = attn(x, wq, wk, wv)
    y = attn(x.to("tvarant"), wq.to("tvarant"), wk.to("tvarant"), wv.to("tvarant"))
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), y_cpu, atol=1e-4)
