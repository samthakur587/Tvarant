# add

```python
torch.add(input, other, *, alpha=1, out=None) → Tensor
input + other
```

Adds `other`, scaled by `alpha`, to `input`:

$$
\text{out}_i = \text{input}_i + \alpha \times \text{other}_i
$$

## Parameters

- **input** (`Tensor`) – First operand (Tvarant).
- **other** (`Tensor or Number`) – Second operand (broadcasting).
- **alpha** (`Number`) – Multiplier for `other`. Default: `1`.

## Shape

- Input: $(*)$
- Output: broadcast shape of `input` and `other`

## Example

```python
y = torch.add(a, b, alpha=1.0)  # or a + b
```

---

[← All supported ops](index.md)
