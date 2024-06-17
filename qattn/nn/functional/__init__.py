"""Quantized functions."""

import math
from typing import Optional, Union

import torch
from torch.library import Library, impl

from . import dynamic  # noqa: F401
from ._matmul import _matmul
from ._flash_attention import _attention

from qattn.quantize import quantize_global

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256

__all__ = ["attention", "matmul", "gelu"]

_qattn_lib = Library("qattn", "DEF")

_qattn_lib.define(
    "static_attention(Tensor q, Tensor k, Tensor v, float qkv_scale, float out_scale, float? sm_scale=None) -> Tensor"
)
_qattn_lib.define(
    "static_matmul(Tensor input, Tensor other, Tensor input_scale, Tensor other_scale, Tensor output_scale, Tensor? bias=None, bool? input_per_channel=False, bool? output_per_channel=False) -> Tensor"
)
_qattn_lib.define(
    "static_linear(Tensor input, Tensor other, float input_scale, float other_scale, "
    "float output_scale, Tensor bias) -> Tensor"
)

_qattn_lib.define(
    "static_linear_per_channel(Tensor input, Tensor other, Tensor input_scale, Tensor other_scale, "
    "Tensor output_scale, Tensor bias, bool input_per_channel, bool weight_per_channel) -> Tensor"
)
_qattn_lib.define("static_gelu(Tensor input, Tensor scale) -> Tensor")


@impl(_qattn_lib, "static_attention", "Meta")
def _attention_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_scale: Union[torch.Tensor, float],
    out_scale: Union[torch.Tensor, float],
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@impl(_qattn_lib, "static_attention", "CUDA")
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_scale: Union[torch.Tensor, float],
    out_scale: Union[torch.Tensor, float],
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Invoke statically quantized attention.

    By default the tensors should be quantized to INT8.
    If they are not then they are quantized to INT8 using qkv_scale.

    This function implements a static quantized flash attention.
    Where the inputs are quantized INT8 tensors.
    Within the kernel, the first dot product is done in integer arithmetic and dequantized
    to floating-point for calculating softmax.
    The output is by default in INT8.

    .. math::

        S = {\\cal Q}(Q) \\cdot {\\cal Q}(K^T) \\, \\frac{s_{Q} \\, s_{K}}{\\sqrt{d}},

        P = \\textrm{softmax}(S),

        O^{\\, \\scriptsize \\textrm{S}} = {\\cal Q}(P \\cdot ({\\cal Q}(V) \\, q_v)).

    Args:
        q (torch.Tensor): INT8 query quantized tensor
        k (torch.Tensor): INT8 key quantized tensor
        v (torch.Tensor): INT8 value quantized tensor
        qkv_scale (Union[torch.Tensor, float]): QKV quantization scale parameter.
        out_scale (Union[torch.Tensor, float]): Output quantization scale.
        sm_scale (float, optional): Softmax scale. Defaults to None.

    Returns:
        torch.Tensor: Result of attention in INT8.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    if q.dtype != torch.int8:
        q = quantize_global(q, qkv_scale, torch.int8)
        k = quantize_global(k, qkv_scale, torch.int8)
        v = quantize_global(v, qkv_scale, torch.int8)

    out = _attention.apply(
        q if q.is_contiguous() else q.contiguous(),
        k if k.is_contiguous() else k.contiguous(),
        v if v.is_contiguous() else v.contiguous(),
        sm_scale,
        qkv_scale,
        out_scale,
    )
    return out


@impl(_qattn_lib, "static_matmul", "Meta")
def _matmul_meta(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: Union[torch.Tensor, float],
    other_scale: Union[torch.Tensor, float],
    output_scale: Union[torch.Tensor, float],
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    other_per_channel: bool = False,
):
    assert input.shape[1] == other.shape[0], "incompatible dimensions"
    M, K = input.shape
    _, N = other.shape
    c = torch.empty((M, N), device=input.device, dtype=torch.int8)
    return c


@impl(_qattn_lib, "static_matmul", "CUDA")
def matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: Union[torch.Tensor, float],
    other_scale: Union[torch.Tensor, float],
    output_scale: Union[torch.Tensor, float],
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    other_per_channel: bool = False,
) -> torch.Tensor:
    """Matrix product of two quantized tensors.

    Args:
        input (torch.Tensor): the first quantized tensor to be multiplied.
        other (torch.Tensor): the second qunatized tensor to be multiplied.
        input_scale (Union[torch.Tensor, float]): the first tensor quantized scale.
        other_scale (Union[torch.Tensor, float]): the second tensor quantized scale.
        output_scale (Union[torch.Tensor, float]): output quantized scale.
        bias (Optional[torch.Tensor]): optional bias.

    Returns:
        torch.Tensor: Quantized output tensor.
    """
    return _matmul.apply(
        input,
        other,
        input_scale,
        other_scale,
        output_scale,
        bias,
        input_per_channel,
        other_per_channel,
    )


@impl(_qattn_lib, "static_linear", "Meta")
def _linear_meta(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: float,
    other_scale: float,
    output_scale: float,
    bias: Optional[torch.Tensor] = None,
):
    input_orig_shape = input.shape
    input = input.view(-1, input_orig_shape[-1])
    out = _matmul_meta(input, other.T, input_scale, other_scale, output_scale, bias)
    out = out.view(*input_orig_shape[:-1], -1)
    return out


@impl(_qattn_lib, "static_linear_per_channel", "Meta")
def _linear_meta_per_channel(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: torch.Tensor,
    other_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    weight_per_channel: bool = False,
):
    input_orig_shape = input.shape
    input = input.view(-1, input_orig_shape[-1])
    out = _matmul_meta(input, other.T, input_scale, other_scale, output_scale, bias)
    out = out.view(*input_orig_shape[:-1], -1)
    return out


@impl(_qattn_lib, "static_linear_per_channel", "CUDA")
def linear_per_channel(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: torch.Tensor,
    other_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    weight_per_channel: bool = False,
):
    input_orig_shape = input.shape
    if input_per_channel:
        if input_scale.dim() == 1:
            input_scale = input_scale.repeat(input_orig_shape[0])
        output_scale = output_scale.repeat(input_scale.shape[0])
    if not input_per_channel:
        input_scale = torch.tensor(input_scale, device=input.device)
        output_scale = torch.tensor(output_scale, device=input.device)
    if not weight_per_channel:
        other_scale = torch.tensor(other_scale, device=input.device)

    return _linear_impl(
        input,
        other,
        input_scale,
        other_scale,
        output_scale,
        bias,
        input_per_channel=input_per_channel,
        weight_per_channel=weight_per_channel,
    )


@impl(_qattn_lib, "static_linear", "CUDA")
def linear(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: float,
    other_scale: float,
    output_scale: float,
    bias: Optional[torch.Tensor] = None,
):
    input_scale = torch.tensor(input_scale, device=input.device)
    other_scale = torch.tensor(other_scale, device=input.device)
    output_scale = torch.tensor(output_scale, device=input.device)
    return _linear_impl(input, other, input_scale, other_scale, output_scale, bias)


def _linear_impl(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: torch.Tensor,
    other_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    weight_per_channel: bool = False,
):
    input_orig_shape = input.shape
    input = input.view(-1, input_orig_shape[-1])
    out: torch.Tensor = _matmul.apply(
        input,
        other.T,
        input_scale,
        other_scale,
        output_scale,
        bias,
        input_per_channel,
        weight_per_channel,
    )
    out = out.view(*input_orig_shape[:-1], -1)
    return out
