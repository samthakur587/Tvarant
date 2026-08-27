# relu

```python
torch.relu(input) → Tensor
torch.nn.functional.relu(input, inplace=False) → Tensor
```

Applies the rectified linear unit function element-wise:

$$
\text{ReLU}(x) = \max(0, x)
$$

## Parameters

- **input** (`Tensor`) – Input tensor on `device="tvarant"`.

## Shape

- Input: $(*)$
- Output: $(*)$, same as input

## Example

```python
y = torch.relu(x)
```

---

[← All supported ops](index.md)
