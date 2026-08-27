# softmax

```python
torch.nn.functional.softmax(input, dim=None, dtype=None) → Tensor
```

Applies softmax along `dim` on Tvarant tensors.

## Parameters

- **input** (`Tensor`) – Input tensor.
- **dim** (`int`) – Dimension along which softmax is computed.

## Shape

- Input: $(*)$
- Output: $(*)$

## Example

```python
p = torch.nn.functional.softmax(logits, dim=-1)
```

---

[← All supported ops](index.md)
