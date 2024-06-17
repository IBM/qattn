import torch
import torch.nn as nn
import pytest
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from qattn.pt2e.quantizer import (
    QAttnQuantizer,
    get_default_qattn_quantization_config,
)


@torch.inference_mode()
def lower_model(model: nn.Module, sample: torch.Tensor, is_dynamic: bool):
    qconfig = get_default_qattn_quantization_config(
        weight_per_channel=False,
        is_dynamic=is_dynamic,
        activation_per_channel=False,
    )

    quantizer = QAttnQuantizer()
    quantizer.set_module_type(torch.nn.Linear, qconfig).set_module_type(
        torch.nn.functional.scaled_dot_product_attention, qconfig
    )
    exported_model = capture_pre_autograd_graph(model, (sample,))
    prepared_model = prepare_pt2e(exported_model, quantizer)
    _ = prepared_model(sample)
    converted_model = convert_pt2e(
        prepared_model,
        fold_quantize=True,
    )
    return converted_model


@pytest.mark.parametrize("is_dynamic", [False, True])
def test_lower_mlp(mlp: nn.Sequential, is_dynamic: bool, mocker):
    from qattn.pt2e.dynamo.backend import backends

    x = torch.randn(32, 16, device="cuda")
    mlp = mlp.to("cuda")

    spy = mocker.spy(backends, "qattn_backend")
    lower_call = mocker.spy(backends, "_lower_linear")

    lowered_model = lower_model(mlp, x, is_dynamic=is_dynamic)
    lowered_model = torch.compile(lowered_model, backend=backends.qattn_backend)

    _ = lowered_model(x)
    _ = lowered_model(x)

    assert spy.call_count == 1
    assert lower_call.call_count == 2


@pytest.mark.parametrize("is_dynamic", [False, True])
def test_lower_attention(attention_block, is_dynamic: bool, mocker):
    from qattn.pt2e.dynamo.backend import backends

    x = torch.randn(3, 197, 192, device="cuda")
    attention_block = attention_block.to("cuda")

    spy = mocker.spy(backends, "qattn_backend")
    lower_call = mocker.spy(backends, "_lower_sdpa")

    lowered_model = lower_model(attention_block, x, is_dynamic=is_dynamic)
    lowered_model = torch.compile(lowered_model, backend=backends.qattn_backend)

    _ = lowered_model(x)
    _ = lowered_model(x)

    assert spy.call_count == 1
    assert lower_call.call_count == 1
