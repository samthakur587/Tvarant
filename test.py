import torch
import torch_tvarant  # registers device "tvarant"

x = torch.randn(4, 8, device="tvarant")
w = torch.randn(8, 8, device="tvarant")
y = torch.nn.functional.relu(x @ w)
print(y)
print(y.device.type)
print(torch.relu(x.cpu() @ w.cpu()))
print(torch.allclose(y.cpu(), torch.relu(x.cpu() @ w.cpu()), atol=1e-4))
assert y.device.type == "tvarant"
assert torch.allclose(y.cpu(), torch.relu(x.cpu() @ w.cpu()), atol=1e-4)

print("Hello, World!")