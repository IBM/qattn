import pytest
import torch
import torch.nn as nn

from qattn.nn import functional as FQ
from qattn.nn.functional import _reference
from qattn.quantize import (
    dequantize,
    find_scale,
    quantize_dynamic_global,
    quantize_global,
)

from ..utils import CUDA_AVAILABLE


@pytest.fixture(params=(197, 256))
def inputs(request):
    return tuple(
        [
            torch.empty(100, 12, request.param, 64, device="cuda").normal_(-0.1022, 1.4821).to(dtype=torch.bfloat16)
            for _ in range(3)
        ]
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_int8_dynamic_attention(inputs):
    scale = 0.125
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(
            enable_mem_efficient=False,
            enable_math=False,
        ):
            torch_out = nn.functional.scaled_dot_product_attention(*inputs, scale=scale)
        triton_out = FQ.dynamic.attention(*inputs, scale)
        ref_out = _reference.dynamic_attention(*inputs, scale)
    assert torch.nn.functional.mse_loss(ref_out.to("cuda:0", dtype=torch.bfloat16), triton_out) < 1e-2
    assert torch.nn.functional.mse_loss(torch_out, triton_out) < 1e-2


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_int8_static_attention(inputs):
    scale = 0.125
    input_dtype = inputs[0].dtype
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(
            enable_mem_efficient=False,
            enable_math=False,
        ):
            torch_out = nn.functional.scaled_dot_product_attention(*inputs, scale=scale)
        ref_out, intermediate = _reference.attention(*inputs, scale, dtype=input_dtype)

        q, k, v = inputs
        qk, k_scale = quantize_dynamic_global(k, -128, 127, torch.int8)
        qq = quantize_global(q, k_scale, torch.int8)
        qv = quantize_global(v, k_scale, torch.int8)
        out_scale = find_scale(intermediate["out"], -128, 127)
        ref_static_out = (
            _reference.static_attention(qq, qk, qv, k_scale, out_scale, scale).to(dtype=input_dtype, device="cuda")
            * out_scale
        )
        triton_out = FQ.attention(
            qq,
            qk,
            qv,
            k_scale,
            out_scale,
            scale,
        )
        triton_out = dequantize(triton_out, out_scale)
    assert torch.nn.functional.mse_loss(torch_out, triton_out) < 1e-2
    assert torch.nn.functional.mse_loss(ref_static_out, triton_out) < 1e-2
    assert torch.nn.functional.mse_loss(ref_out, triton_out) < 1e-2
