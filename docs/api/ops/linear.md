# linear

```python
torch.nn.functional.linear(input, weight, bias=None) → Tensor
```

Applies an affine linear transformation with a **transposed** weight layout matching `nn.Linear`:

$$
y = x A^T + b
$$

## Parameters

- **input** (`Tensor`) – Input of shape $(*, in\_features)$.
- **weight** (`Tensor`) – Weight of shape $(out\_features, in\_features)$.
- **bias** (`Tensor, optional`) – Bias of shape $(out\_features)$.

## Shape

- Input: $(*, in)$
- Output: $(*, out)$

## Example

```python
y = torch.nn.functional.linear(x, w, b)
```

---

[← All supported ops](index.md)
