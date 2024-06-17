"""Dynamically quantized functions."""

from typing import Optional, Union
import torch
from torch.library import Library, impl

from ._matmul import _matmul  # noqa: F401

_qattn_lib = Library("qattn", "FRAGMENT")


_qattn_lib.define(
    "dynamic_linear(Tensor input, Tensor other, float input_scale, float other_scale, Tensor? bias=None) -> Tensor",
)
_qattn_lib.define(
    "dynamic_linear_per_channel(Tensor input, Tensor other, Tensor input_scale, "
    "Tensor other_scale, Tensor? bias, bool? input_per_channel=False, bool? weight_per_channel=False) -> Tensor"
)


_qattn_lib.define(
    "dynamic_matmul(Tensor input, Tensor other, "
    "Tensor input_scale, Tensor other_scale, Tensor? bias=None, "
    "bool? input_per_channel=False, bool? output_per_channel=False) -> Tensor",
)


@impl(_qattn_lib, "dynamic_matmul", "Meta")
def _matmul_meta(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: Union[torch.Tensor, float],
    other_scale: Union[torch.Tensor, float],
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    other_per_channel: bool = False,
):
    assert input.shape[1] == other.shape[0], "incompatible dimensions"
    M, K = input.shape
    _, N = other.shape
    c = torch.empty((M, N), device=input.device, dtype=input.dtype)
    return c


@impl(_qattn_lib, "dynamic_matmul", "CUDA")
def matmul(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: Union[torch.Tensor, float],
    other_scale: Union[torch.Tensor, float],
    bias: Optional[torch.Tensor] = None,
    input_per_channel=False,
    other_per_channel=False,
) -> torch.Tensor:
    """Dynamically quantized matrix multiplication.

    Args:
        input (torch.Tensor): the first tensor to be multiplied.
        other (torch.Tensor): the second tensor to be multiplied.
        input_scale (Union[torch.Tensor, float]): the first tensor quantized scale.
        other_scale (Union[torch.Tensor, float]): the second tensor quantized scale
        bias (Optional[torch.Tensor], optional): optional bias. Defaults to None.

    Returns:
        torch.Tensor: the output tensor.
    """
    return _matmul.apply(
        input,
        other,
        input_scale,
        other_scale,
        bias,
        input_per_channel,
        other_per_channel,
    )


@impl(_qattn_lib, "dynamic_linear", "Meta")
def _linear_meta(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: float,
    other_scale: float,
    bias: Optional[torch.Tensor] = None,
):
    input_orig_shape = input.shape
    input = input.view(-1, input_orig_shape[-1])
    out = _matmul_meta(input, other.T, input_scale, other_scale, bias)
    out = out.view(*input_orig_shape[:-1], -1)
    return out


@impl(_qattn_lib, "dynamic_linear_per_channel", "Meta")
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


@impl(_qattn_lib, "dynamic_linear", "CUDA")
def dynamic_linear(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: float,
    other_scale: float,
    bias: Optional[torch.Tensor] = None,
):
    input_scale = torch.tensor(input_scale, device=input.device)
    other_scale = torch.tensor(other_scale, device=input.device)
    return _dynamic_linear_impl(input, other, input_scale, other_scale, bias, False, False)


@impl(_qattn_lib, "dynamic_linear_per_channel", "CUDA")
def dynamic_linear_per_channel(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: torch.Tensor,
    other_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    weight_per_channel: bool = False,
):
    return _dynamic_linear_impl(
        input,
        other,
        input_scale,
        other_scale,
        bias,
        input_per_channel,
        weight_per_channel,
    )


def _dynamic_linear_impl(
    input: torch.Tensor,
    other: torch.Tensor,
    input_scale: Union[torch.Tensor, float],
    other_scale: Union[torch.Tensor, float],
    bias: Optional[torch.Tensor] = None,
    input_per_channel: bool = False,
    weight_per_channel: bool = False,
):
    input_orig_shape = input.shape
    input = input.view(-1, input_orig_shape[-1])
    if input_per_channel:
        if input_scale.dim() == 1:
            input_scale = input_scale.repeat(input_orig_shape[0])

    out: torch.Tensor = _matmul.apply(
        input,
        other.T,
        input_scale,
        other_scale,
        bias,
        input_per_channel,
        weight_per_channel,
    )
    out = out.view(*input_orig_shape[:-1], -1)
    return out
