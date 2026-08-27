# Supported ops

ATen and custom ops registered for `PrivateUse1` (`device="tvarant"`).
Unlisted ops **fall back to CPU** (tensors are copied as needed).

Click an op for signature, parameters, shape, and examples.

## Factories & memory

| Op | Page |
|---|---|
| `empty` / `empty_strided` | [empty](empty.md) |
| `zeros` / `ones` / `full` | [zeros / ones / full](zeros.md) |
| `clone` / `contiguous` | [clone / contiguous](clone.md) |

## Pointwise

| Op | Page |
|---|---|
| `add` | [add](add.md) |
| `mul` | [mul](mul.md) |
| `sub` | [sub](sub.md) |
| `relu` | [relu](relu.md) |
| `silu` | [silu](silu.md) |
| `neg` / `abs` / `exp` / `sqrt` / … | [unary](neg.md) |

## Linear algebra & NN

| Op | Page |
|---|---|
| `mm` / `addmm` / `matmul` / `bmm` | [mm family](mm.md) |
| `linear` | [linear](linear.md) |
| `softmax` | [softmax](softmax.md) |
| `layer_norm` | [layer_norm](layer_norm.md) |
| `embedding` | [embedding](embedding.md) |

## Custom (`torch.ops.tvarant`)

| Op | Page |
|---|---|
| `linear_act` | [linear_act](linear_act.md) |
| `pointwise` | [pointwise](pointwise.md) |
