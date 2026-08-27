# mm / addmm / matmul / bmm

```python
torch.mm(input, mat2) → Tensor
torch.addmm(input, mat1, mat2, *, beta=1, alpha=1) → Tensor
torch.matmul(input, other) → Tensor
torch.bmm(input, mat2) → Tensor
```

Matrix products. `mm`/`addmm` use the fused GEMM path; `matmul`/`bmm` cover 2D and batched cases.

## Parameters

- **input / mat1** (`Tensor`) – Left matrix / batch.
- **mat2 / other** (`Tensor`) – Right matrix / batch.
- **beta, alpha** (`Number`) – `addmm`: `beta*input + alpha*(mat1@mat2)`.

## Shape

- `mm`: $(n,m) \times (m,p) \rightarrow (n,p)$
- `bmm`: $(B,n,m) \times (B,m,p) \rightarrow (B,n,p)$

## Example

```python
y = torch.mm(a, b)
```

---

[← All supported ops](index.md)
