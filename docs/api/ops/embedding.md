# embedding

```python
torch.nn.functional.embedding(input, weight, ...) → Tensor
```

Lookup embeddings. Indices should be convertible to int32 on the Tvarant path.

## Parameters

- **input** (`LongTensor`) – Token indices.
- **weight** (`Tensor`) – Embedding table `(V, D)` on Tvarant.

## Shape

- Input: $(*)$
- Output: $(*, D)$

## Example

```python
e = torch.nn.functional.embedding(tokens, weight)
```

---

[← All supported ops](index.md)
