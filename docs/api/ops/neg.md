# neg / abs / exp / sqrt

```python
torch.neg(input) → Tensor
torch.abs(input) → Tensor
torch.exp(input) → Tensor
torch.exp2(input) → Tensor
torch.sqrt(input) → Tensor
torch.rsqrt(input) → Tensor
```

Unary element-wise math ops on fp32 Tvarant tensors (host + OpenCL kernels).

## Parameters

- **input** (`Tensor`) – Input tensor.

## Shape

- Input: $(*)$
- Output: $(*)$

## Example

```python
y = torch.exp(x)
z = torch.rsqrt(torch.abs(x) + 1e-6)
```

---

[← All supported ops](index.md)
