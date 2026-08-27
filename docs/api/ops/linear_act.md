# torch.ops.tvarant.linear_act

```python
torch.ops.tvarant.linear_act(x, weight, bias, act, trans_b=False) → Tensor
```

Fused GEMM + optional bias + activation. Used by the graph compiler for Linear+ReLU/SiLU epilogues.

## Parameters

- **x** (`Tensor`) – Activations `(..., K)`.
- **weight** (`Tensor`) – Weight matrix.
- **bias** (`Tensor or None`) – Optional bias.
- **act** (`str`) – `"none"`, `"relu"`, or `"silu"`.
- **trans_b** (`bool`) – If `True`, treat weight like `nn.Linear` (`A^T`). Default: `False`.

## Shape

- Output: `(..., N)` where `N` is the output feature size

## Example

```python
y = torch.ops.tvarant.linear_act(x, w, b, 'relu', True)
```

---

[← All supported ops](index.md)
