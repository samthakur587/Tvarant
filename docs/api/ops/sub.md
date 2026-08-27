# sub

```python
torch.sub(input, other, *, alpha=1, out=None) → Tensor
input - other
```

Subtracts `alpha * other` from `input` (implemented via negated `add` when registered).

## Parameters

- **input** (`Tensor`) – Minuend.
- **other** (`Tensor or Number`) – Subtrahend.
- **alpha** (`Number`) – Multiplier for `other`. Default: `1`.

## Shape

- Output: broadcast shape

## Example

```python
y = a - b
```

---

[← All supported ops](index.md)
