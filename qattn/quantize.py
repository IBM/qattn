"""Quantization primitives.

This module implements quantization primitives like quantize, find_scale, and dequantize.
"""

from typing import Union, Tuple, Optional
import torch
from torch.library import Library, impl


QUANTIZATION_RANGES = {
    torch.int8: (-128, 127),
    torch.int16: (-(2 << 14), (2 << 14) - 1),
    torch.int32: (-(2 << 30), (2 << 30) - 1),
}

quantized_decomposed_lib = Library("quantized_decomposed", "IMPL")


@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "CUDA")
def choose_qparams_symmetric_tensor(
    input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given an input tensor calcuate the quantization parameters for INT8.

    This function overrides pytorch implementation to support FP32/16/BF16.

    Args:
        input (torch.Tensor): floating point input Tensor
        qmin (int): minimum quantized value for the target Tensor
        qmax (int): maximum quantized value for the target Tensor
        dtype (torch.dtype): Not used as we support only INT8. Used to match signature

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing scale and zero point
    """
    assert input.dtype in {torch.float32, torch.float16, torch.bfloat16}
    scale = find_scale(input, qmin, qmax, tuple(), False, eps)
    zero_point = torch.zeros_like(scale, dtype=torch.int8)
    return scale, zero_point


def find_scale(
    x: torch.Tensor,
    qmin: int,
    qmax: int,
    dim: Union[int, Tuple[int, ...]] = tuple(),
    keepdim: bool = False,
    eps=1.1920928955078125e-07,
) -> Union[torch.Tensor, float]:
    """Find quantization scale factor for a tensor given quantization ranges.

    Args:
        x (torch.Tensor): Floating-point tensor.
        qmin (int): Quantization minimum range.
        qmax (int): Quantization maximum range.

    Returns:
        Union[torch.Tensor, float]: Quantization scaling factor.
    """
    max_val = torch.amax(torch.abs(x), dim=dim, keepdim=keepdim)
    scale = max_val / ((qmax - qmin) / 2)
    eps = torch.tensor(eps, device=scale.device)
    scale = torch.max(scale, eps)
    return scale


def _quantize_inner(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype: torch.dtype,
):
    xq = torch.clamp(torch.round(x / scale), qmin, qmax).to(dtype)
    return xq


def quantize_dynamic_global(
    x: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize given tensor to quantization range and data type.

    Args:
        x (torch.Tensor): Floating-point tensor.
        qmin (int): Quantization minimum range.
        qmax (int): Quantization maximum range.
        dtype (torch.dtype): Quantized tensor dtype.

    Returns:
        Tuple[torch.Tensor, Union[torch.Tensor, float]]: Tuple with quantized tensor and scaling factor.
    """
    scale = find_scale(x, qmin=qmin, qmax=qmax)
    xq = _quantize_inner(x, scale, qmin, qmax, dtype)
    return xq, torch.tensor(scale)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "CUDA")
def quantize_per_tensor(
    x: torch.Tensor,
    scale: float,
    zero_point: int,
    qmin: int,
    qmax: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Symmetric quantization for the floating point Tensor to Quantized Tensor

    Args:
        x (torch.Tensor): floating point input Tensor
        scale (float): quantization scale
        zero_point (int): quantization zero point
        qmin (int): quantization minimum value
        qmax (int): quantization maximum value
        dtype (torch.dtype): quantization data type

    Returns:
        torch.Tensor: Quantized Tensor
    """
    return quantize_global(x, torch.tensor(scale, device=x.device), dtype=dtype)


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "CUDA")
def quantize_per_tensor_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Symmetric quantization for the floating point Tensor to Quantized Tensor

    Args:
        x (torch.Tensor): floating point input Tensor
        scale (float): quantization scale
        zero_point (int): quantization zero point
        qmin (int): quantization minimum value
        qmax (int): quantization maximum value
        dtype (torch.dtype): quantization data type

    Returns:
        torch.Tensor: Quantized Tensor
    """
    return quantize_global(x, scale, dtype=dtype)


def quantize_global(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Quantize the tensor given quantization scale to given data type.

    Args:
        x (torch.Tensor): Floating-point tensor.
        scale (torch.Tensor): Quantization scale factor.
        dtype (torch.dtype): Quantized data type. Based on it the quantization ranges will be selected.
            Supported formats are int8, int16, and int32

    Returns:
        torch.Tensor: Quantized torch tensor.
    """

    qmin, qmax = QUANTIZATION_RANGES[dtype]
    return _quantize_inner(x, scale, qmin, qmax, dtype)


def quantize_per_channel(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Symmetric quantization of the input tensor to the quantized tensor per channel.

    Args:
        x (torch.Tensor): the floating point input tensor.
        scale (torch.Tensor): the scale quantization parameter tensor
        qmin (int): quantization minimum value
        qmax (int): quantization maximum value
        dtype (torch.dtype): quantization data type

    Returns:
        torch.Tensor: quantized tensor
    """
    if scale.ndim == 1:
        scale = scale.view(-1, 1)
    x = _quantize_inner(x, scale, qmin, qmax, dtype)
    return x


def choose_qparams_per_channel_symmetric(
    x: torch.Tensor,
    qmin: int,
    qmax: int,
    eps: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose the symmetric quantization parameters per channel.

    Args:
        x (torch.Tensor): the floating input tensor
        qmin (int): quantization minimum value
        qmax (int): quantization maximum value

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantization parameters tuple scale and zero point
    """
    scale = find_scale(x, qmin, qmax, dim=1)
    zp = torch.zeros_like(scale, dtype=torch.int8)
    return scale, zp


@impl(quantized_decomposed_lib, "quantize_per_channel", "CUDA")
def _quantize_per_channel_impl(
    x: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    axis: int,
    qmin: int,
    qmax: int,
    dtype: torch.dtype,
):
    """Quantize per channel floating input tensors.

    This implementation overrides torch.ops.quantized_decomposed_lib.quantize_per_channel

    Args:
        x (torch.Tensor): the floating point input tensor
        scales (torch.Tensor): quantization parameter - scale per channel
        zero_points (torch.Tensor): quantization parameter - zero points; currently not used.
        axis (int): _description_
        qmin (int): _description_
        qmax (int): _description_
        dtype (torch.dtype): _description_

    Returns:
        _type_: _description_
    """
    return quantize_per_channel(x, scales, qmin, qmax, dtype)


def quantize_dynamic_per_channel(
    x: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype: torch.dtype,
    dim=2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = find_scale(x, qmin=qmin, qmax=qmax, dim=dim, keepdim=True)
    x = _quantize_inner(x, scale, qmin, qmax, dtype)
    return x, scale.to(torch.float32)


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CUDA")
def dequantize_per_tensor(
    x: torch.Tensor,
    scale: float,
    zero_point: int,
    qmin: int,
    qmax: int,
    dtype=torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
):
    if out_dtype is None:
        out_dtype = torch.bfloat16
    return dequantize(x, scale, dtype=out_dtype)


@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "CUDA")
def dequantize_per_tensor_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype=torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
):
    if out_dtype is None:
        out_dtype = torch.bfloat16
    return dequantize(x, scale, dtype=out_dtype)


def dequantize(xq: torch.Tensor, scale: Union[torch.Tensor, float], dtype: torch.dtype = torch.half) -> torch.Tensor:
    """Dequantize quantized tensor to floating point tensor.

    Args:
        xq (torch.Tensor): Quantized tensor.
        scale (Union[torch.Tensor, float]): Quantization scaling factor.
        dtype (torch.dtype, optional): Target data type.. Defaults to torch.half.

    Returns:
        torch.Tensor: Dequantized floating-point tensor.
    """
    if isinstance(scale, torch.Tensor):
        scale = scale.view(-1, 1)
    return (xq * scale).to(dtype)


@impl(quantized_decomposed_lib, "dequantize_per_channel", "CUDA")
def dequantize_per_channel(
    x: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
):
    if out_dtype is None:
        out_dtype = torch.bfloat16
    return dequantize(x, scales, out_dtype)
