# Getting started

After [installing](install.md) `torch_tvarant`, verify the device and run a small model.

## Verify installation

```python
import torch
import torch_tvarant

print(torch.tvarant.is_available())   # True
print(torch.tvarant.backend())        # 'sim' or 'opencl'
x = torch.ones(4, device="tvarant")
print(x.device)                       # tvarant:0
```

## Run a small MLP

```python
import torch
import torch.nn as nn
import torch_tvarant

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
).to("tvarant")

x = torch.randn(2, 16, device="tvarant")
y = model(x)
print(y.shape)  # torch.Size([2, 4])
```

## Compile for inference

```python
import torch_tvarant

compiled = torch_tvarant.compiler.compile(model)
y = compiled(x)
```

See [JIT Compiler](jit-compiler.md) for fusion details.

## Learn more

| Topic | Page |
|---|---|
| Device helpers | [torch.tvarant](api/torch.md) |
| Supported ops | [Supported ops](api/ops.md) |
| Kernels | [Kernels](api/kernels.md) |
| C++ extension | [C++ API](api/cpp.md) |
| FPGA path | [FPGA / OpenCL](fpga-opencl.md) |
