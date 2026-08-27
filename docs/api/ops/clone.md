# clone / contiguous

```python
Tensor.clone(memory_format=torch.preserve_format) → Tensor
Tensor.contiguous(memory_format=torch.contiguous_format) → Tensor
```

Materialize a copy (`clone`) or a contiguous storage (`contiguous`) on Tvarant via `_copy_from`.

## Parameters

- **memory_format** (`torch.memory_format`) – Desired memory format.

## Shape

- Output: same logical shape as input

## Example

```python
y = x.clone()
c = x.t().contiguous()
```

---

[← All supported ops](index.md)
