# silu

```python
torch.nn.functional.silu(input, inplace=False) → Tensor
```

Applies the Sigmoid Linear Unit (SiLU / Swish):

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

## Parameters

- **input** (`Tensor`) – Input tensor.

## Shape

- Input: $(*)$
- Output: $(*)$

## Example

```python
y = torch.nn.functional.silu(x)
```

---

[← All supported ops](index.md)
