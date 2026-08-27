# empty

```python
torch.empty(*size, *, dtype=None, layout=torch.strided, device=None, pin_memory=False, memory_format=torch.contiguous_format) → Tensor
```

Returns a tensor filled with uninitialized data on `device="tvarant"`.

## Parameters

- **size** (`int...`) – Size of the returned tensor.
- **dtype** (`torch.dtype, optional`) – Desired data type. Defaults to the global default (prefer `float32`).
- **device** (`torch.device, optional`) – Use `"tvarant"` / `"tvarant:0"`.
- **memory_format** (`torch.memory_format, optional`) – Memory format for the result.

## Shape

- Output: $(\textit{size})$

## Example

```python
x = torch.empty(2, 3, device='tvarant', dtype=torch.float32)
```

---

[← All supported ops](index.md)
