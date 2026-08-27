# layer_norm

```python
torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05) → Tensor
```

Applies Layer Normalization over the last dimensions described by `normalized_shape` (via `native_layer_norm`).

## Parameters

- **input** (`Tensor`) – Input tensor.
- **normalized_shape** (`list of int`) – Shape over which to normalize.
- **weight, bias** (`Tensor, optional`) – Affine parameters.
- **eps** (`float`) – Numerical stability epsilon.

## Shape

- Output: same as input

## Example

```python
y = torch.nn.functional.layer_norm(x, x.shape[-1:])
```

---

[← All supported ops](index.md)
