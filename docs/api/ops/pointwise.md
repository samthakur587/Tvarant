# torch.ops.tvarant.pointwise

```python
torch.ops.tvarant.pointwise(inputs, ops, a, b, input_ids, alphas, consts) → Tensor
```

Runs a JIT pointwise SSA program over equal-numel Tvarant tensors. Prefer `torch_tvarant.compiler.compile` rather than building programs by hand.

!!! note

    Opcode IDs must match `csrc/jit/Jit.h` (`LOAD`, `ADD`, `MUL`, `RELU`, …).

## Parameters

- **inputs** (`List[Tensor]`) – Input tensors (same numel).
- **ops, a, b, input_ids** (`List[int]`) – SSA node opcodes and operands.
- **alphas, consts** (`List[float]`) – Per-node scalars.

## Shape

- Output: same shape as `inputs[0]`

## Example

```python
# Emitted by torch_tvarant.compiler.fuse_pointwise — not usually called directly.
```

---

[← All supported ops](index.md)
