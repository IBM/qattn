import pytest
import torch

from qattn.nn import functional as FQ
from qattn.nn.functional import _reference
from qattn.quantize import find_scale, quantize_dynamic_global, dequantize, quantize_global

from ..utils import CUDA_AVAILABLE


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No Cuda avaialable")
@pytest.mark.parametrize("x_shape,w_shape", [((128, 128), (128, 128)), ((197, 192), (192, 576))])
def test_dynamic_int8_matmul(x_shape, w_shape):
    torch.rand
    w = torch.randn(w_shape, device="cuda")
    x = torch.randn(x_shape, device="cuda")
    w_int, w_scale = quantize_dynamic_global(w, qmin=-128, qmax=127, dtype=torch.int8)
    x_scale = find_scale(x, qmin=-128, qmax=127)

    ref_output = _reference._dynamic_matmul(
        x,
        w_int,
        x_scale=x_scale,
        w_scale=w_scale,
    ).cuda()
    triton_out = FQ.dynamic.matmul(x, w_int, x_scale, w_scale)
    torch.cuda.synchronize()
    assert torch.allclose(ref_output.cuda(), triton_out, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No Cuda avaialable")
@pytest.mark.parametrize("x_shape,w_shape", [((128, 128), (128, 128)), ((197, 192), (192, 576))])
def test_static_int8_matmul(x_shape, w_shape):
    w = torch.randn(w_shape, device="cuda")
    x = torch.randn(x_shape, device="cuda")
    out = x @ w
    out_scale = find_scale(out, qmin=-128, qmax=127)
    x_int, x_scale = quantize_dynamic_global(x, qmin=-128, qmax=127, dtype=torch.int8)
    w_int, w_scale = quantize_dynamic_global(w, qmin=-128, qmax=127, dtype=torch.int8)
    x_int_fp32 = dequantize(x_int, x_scale, torch.float32)
    w_int_fp32 = dequantize(w_int, w_scale, torch.float32)
    out_ref_fp32 = x_int_fp32 @ w_int_fp32
    out_ref_fp32_int8 = quantize_global(out_ref_fp32, out_scale, torch.int8)

    ref_output = _reference._static_matmul(
        x_int,
        w_int,
        x_scale=x_scale,
        w_scale=w_scale,
        out_scale=out_scale,
    ).cuda()
    triton_out = FQ.matmul(x_int, w_int, x_scale, w_scale, out_scale)
    torch.cuda.synchronize()
    assert torch.equal(triton_out, ref_output)
    assert torch.equal(
        dequantize(triton_out, out_scale, torch.float32), dequantize(ref_output, out_scale, torch.float32)
    )

    assert torch.equal(triton_out, out_ref_fp32_int8)
    assert torch.equal(
        dequantize(triton_out, out_scale, torch.float32), dequantize(out_ref_fp32_int8, out_scale, torch.float32)
    )
