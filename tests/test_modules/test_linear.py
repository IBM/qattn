import pytest
import os
import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfigMapping,
    quantize_fx,
)

from qattn.nn.modules import DynamicQuantizedLinear, QuantizedLinear
from qattn.quantize import quantize_dynamic_global, quantize_dynamic_per_channel
from qattn.fx.qconfig import get_default_qconfig
from qattn.backends_config.qattn import get_qattn_backend_config

from ..utils import CUDA_AVAILABLE


os.environ["TRITON_INTERPRET"] = "1"


@pytest.fixture
def mapping() -> QConfigMapping:
    m = QConfigMapping().set_global(get_default_qconfig())
    return m


def torch_module(dim, mlp_ratio, bias):
    class M(nn.Module):
        def __init__(self, dim, mlp_ratio, bias, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.linear = nn.Linear(dim, dim * mlp_ratio, bias=bias)

        def forward(self, x):
            return self.linear(x)

    return M(dim, mlp_ratio, bias)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("mlp_ratio", [3, 4])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "per_channel",
    [
        {"input_per_channel": False, "per_channel": True},
        {"input_per_channel": True, "per_channel": True},
        False,
    ],
)
def test_linear_dynamic(batch_size, mlp_ratio, bias, per_channel):
    dim = 192
    x = torch.randn((batch_size, 197, dim), dtype=torch.float32, device="cuda:0")
    torch_linear = nn.Linear(dim, dim * mlp_ratio, bias=bias, device="cuda:0")
    if isinstance(per_channel, dict):
        weight_per_channel = per_channel["per_channel"]
        input_per_channel = per_channel["input_per_channel"]
    else:
        weight_per_channel = False
        input_per_channel = False
    linear = DynamicQuantizedLinear.from_float(
        torch_linear,
        per_channel=weight_per_channel,
        input_per_channel=input_per_channel,
    )

    with torch.no_grad():
        if weight_per_channel and input_per_channel:
            xq, x_scale = quantize_dynamic_per_channel(x, -128, 127, dtype=torch.int8)
        else:
            xq, x_scale = quantize_dynamic_global(x, -128, 127, dtype=torch.int8)
        x_scale = x_scale.cpu()
        torch_out = torch.ops.aten.linear.default(
            xq.cpu().to(torch.int32),
            linear.weight.cpu().to(torch.int32),
        )
        if weight_per_channel:
            scale_out = x_scale.cpu() * linear.scale.cpu().view(1, -1)
        else:
            scale_out = x_scale * linear.scale.cpu()
        torch_out = torch_out * scale_out
        if bias:
            torch_out = torch_out + torch_linear.bias.cpu()
        triton_out = linear(x, x_scale.cuda())
    torch.cuda.synchronize()
    assert torch.allclose(torch_out.cuda(), triton_out, atol=1, rtol=1e-1)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("mlp_ratio", [3, 4])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "per_channel",
    [
        {"input_per_channel": False, "per_channel": True},
        {"input_per_channel": True, "per_channel": True},
        False,
    ],
)
def test_linear_static(batch_size, mlp_ratio, bias, per_channel, mapping):
    if per_channel:
        input_per_channel = per_channel.get("input_per_channel", False)
        mapping = mapping.set_global(get_default_qconfig(**per_channel))
    else:
        input_per_channel = False
    dim = 192
    x = torch.randn((batch_size, 197, dim), dtype=torch.float32, device="cuda:0") + 0.5
    module = torch_module(dim, mlp_ratio, bias=bias).eval()
    module = quantize_fx.prepare_fx(
        module, qconfig_mapping=mapping, example_inputs=(x,), backend_config=get_qattn_backend_config()
    )
    with torch.no_grad():
        module(x.cpu())
    module = quantize_fx.convert_to_reference_fx(
        module,
        backend_config=get_qattn_backend_config(),
    )
    linear = QuantizedLinear.from_reference(module.linear, module.linear_scale_0)

    with torch.no_grad():
        if input_per_channel:
            xq, x_scale = quantize_dynamic_per_channel(x.to(torch.bfloat16), -128, 127, dtype=torch.int8)
        else:
            xq, x_scale = quantize_dynamic_global(x, -128, 127, dtype=torch.int8)
        x_scale = x_scale.cpu()
        torch_out = torch.ops.aten.linear.default(
            xq.cpu().to(torch.int32),
            linear.weight.cpu().to(torch.int32),
        )
        if bias:
            bias_scale = x_scale * linear.scale.cpu()
            torch_out = torch_out + torch.round(linear.bias.cpu() / bias_scale).to(torch.int32)
        if per_channel:
            scale_out = (x_scale.cpu() * linear.scale.cpu().view(1, -1)) / linear.out_scale.cpu().view(-1, 1)
        else:
            scale_out = (x_scale * linear.scale.cpu()) / module.linear_scale_0
        torch_out = torch.round(torch_out * scale_out).to(dtype=torch.int8, device="cuda")
        triton_out = linear(xq, x_scale.cuda())
    torch.cuda.synchronize()
    assert torch.equal(torch_out, triton_out)
