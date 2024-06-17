import torch
from torch.ao.quantization import QConfigMapping, quantize_fx
import torch.nn as nn
import pytest

import qattn
import qattn.nn as qnn
from qattn.backends_config import get_qattn_backend_config


@pytest.fixture
def model() -> nn.Module:
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 64)
            self.qkv = nn.Linear(64, 576)

        def forward(self, x):
            B, N = x.shape[:2]
            x = self.linear(x)
            qkv = self.qkv(x).reshape(B, N, 3, 3, 64).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            x = nn.functional.scaled_dot_product_attention(q, k, v)
            return x

    m = M()
    return m


@pytest.fixture(params=(True, False))
def qmapping_dynamic(request) -> QConfigMapping:
    dynamic_qconfig = qattn.get_default_qconfig(True)
    mapping = QConfigMapping().set_global(None).set_object_type(nn.Linear, dynamic_qconfig)
    if request.param:
        mapping = mapping.set_object_type(nn.functional.scaled_dot_product_attention, dynamic_qconfig)
    return mapping


@pytest.fixture(params=[True, False])
def qmapping_static(request) -> QConfigMapping:
    qconfig = qattn.get_default_qconfig()
    mapping = QConfigMapping().set_global(None).set_object_type(nn.Linear, qconfig)
    if request.param:
        mapping = mapping.set_object_type(nn.functional.scaled_dot_product_attention, qconfig)
    return mapping


def test_lower_dynamic_quant(model, qmapping_dynamic: QConfigMapping):
    model = quantize_fx.prepare_fx(
        model, qmapping_dynamic, example_inputs=(torch.randn(1, 32, 32),), backend_config=get_qattn_backend_config()
    )
    quant_sdpa = nn.functional.scaled_dot_product_attention in list(qmapping_dynamic.object_type_qconfigs.keys())
    model = qattn.convert(model=model, quant_sdpa=quant_sdpa, is_dynamic=True)

    assert isinstance(getattr(model, "linear"), qnn.DynamicQuantizedLinear)
    if quant_sdpa:
        for n in filter(lambda n: n.op == "call_function" and "attention" in n.name, model.graph.nodes):
            assert n.target != nn.functional.scaled_dot_product_attention


def test_lower_static_quant(model, qmapping_static: QConfigMapping):
    model = quantize_fx.prepare_fx(
        model, qmapping_static, example_inputs=(torch.randn(1, 32, 32),), backend_config=get_qattn_backend_config()
    )
    quant_sdpa = nn.functional.scaled_dot_product_attention in list(qmapping_static.object_type_qconfigs.keys())
    with torch.inference_mode():
        model(torch.randn(1, 32, 32))
    model = qattn.convert(model=model, quant_sdpa=quant_sdpa, is_dynamic=False)

    assert isinstance(getattr(model, "linear"), qnn.QuantizedLinear)
    if quant_sdpa:
        for n in filter(lambda n: n.op == "call_function" and "attention" in n.name, model.graph.nodes):
            assert n.target != nn.functional.scaled_dot_product_attention
