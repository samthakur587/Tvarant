import os

import pytest
import torch
import torch.nn.functional as F
import torch_tvarant  # noqa: F401

pytestmark = pytest.mark.skipif(
    os.environ.get("TVARANT_BACKEND", "sim").lower() not in {"opencl", "fpga"},
    reason="OpenCL/FPGA tests require TVARANT_BACKEND=opencl",
)


def test_opencl_backend_name():
    assert torch_tvarant._C.backend() == "opencl"


def test_opencl_add_relu_mm():
    a = torch.tensor([[1.0, -2.0], [3.0, -4.0]], device="tvarant")
    b = torch.ones(2, 2, device="tvarant")
    y = F.relu(a @ b)
    expected = torch.relu(a.cpu() @ b.cpu())
    assert y.device.type == "tvarant"
    assert torch.allclose(y.cpu(), expected, atol=1e-4)
