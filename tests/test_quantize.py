import torch
import pytest
from qattn import quantize

from .utils import CUDA_AVAILABLE


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_qdq_dynamic():
    x = torch.randn(1, 3, 3, device="cuda")
    xqt, xq_scale = quantize.quantize_dynamic_global(x, -128, 127, torch.int8)
    xq = torch.quantize_per_tensor(x, scale=xq_scale, zero_point=0, dtype=torch.qint8)
    assert torch.allclose(xq.int_repr(), xqt, atol=1)

    xdq = xq.dequantize()
    xdqt = quantize.dequantize(xqt, xq_scale, torch.float32)
    assert torch.allclose(xdq, xdqt)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("dtype,qmin,qmax", ((torch.int8, -128, 127), (torch.int32, -(2 << 30), (2 << 30) - 1)))
def test_qdq_static(dtype, qmin, qmax):
    x = torch.randn(1, 3, 3, device="cuda")
    xq_scale = quantize.find_scale(x, qmin, qmax)
    qdtype = torch.qint8 if dtype == torch.int8 else torch.qint32
    xq = torch.quantize_per_tensor(x, scale=xq_scale, zero_point=0, dtype=qdtype)
    xqt = quantize.quantize_global(x, xq_scale, dtype)
    assert torch.allclose(xq.int_repr(), xqt, atol=1)

    xdq = xq.dequantize()
    xdqt = quantize.dequantize(xqt, xq_scale, torch.float32)
    assert torch.allclose(xdq, xdqt)
