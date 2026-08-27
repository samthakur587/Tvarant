# zeros / ones / full

```python
torch.zeros(*size, *, dtype=None, device=None, ...) → Tensor
torch.ones(*size, *, dtype=None, device=None, ...) → Tensor
torch.full(size, fill_value, *, dtype=None, device=None, ...) → Tensor
```

Factory ops that allocate on Tvarant then fill via `fill_`. `*_like` variants match another tensor's shape/device.

!!! note

    On older builds without PrivateUse1 factory impls, factories may allocate via empty+fill after the extended-ops merge.

## Parameters

- **size** (`int... or size`) – Output shape.
- **fill_value** (`Number`) – Constant for `full` / `full_like`.
- **dtype** (`torch.dtype, optional`) – Desired dtype (fp32 path is fully accelerated).

## Shape

- Output: same as `size` / like-tensor

## Example

```python
z = torch.zeros(2, 3, device='tvarant')
o = torch.ones_like(z)
f = torch.full((2, 3), 2.5, device='tvarant')
```

---

[← All supported ops](index.md)
