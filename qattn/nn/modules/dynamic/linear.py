"""Dynamically quantzied linear layer"""

from typing import Optional
import torch
import torch.nn as nn

from qattn.nn.functional.dynamic import matmul
from qattn.quantize import find_scale, quantize_dynamic_global, quantize_dynamic_per_channel


class DynamicQuantizedLinear(nn.Linear):
    """Dynamically quantized linear layer.

    Args:
        scale (torch.Tensor): quantized weight scale factor
        in_features (int): size of each input sample
        out_features(int): size of each output sample
        bias (bool): If set to `False`, the layer will not learn an additive bias. Default: `True`

    """

    scale: torch.Tensor
    _per_channel: bool = False
    _input_per_channel: bool = False

    def forward(
        self,
        input: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Defines the forward computation of linear layer.

        If scale factor is not provided it will be calculated based on the input tensor.

        Args:
            input (torch.Tensor): input torch tensor.
            input_scale (torch.Tensor, optional): input quantization scale factor. Defaults to None.

        Returns:
            torch.Tensor: output tensor.
        """
        input_shape = input.shape
        input_per_channel = self._input_per_channel
        if input_scale is None:
            if input.ndim == 2 or not input_per_channel:
                input_scale = find_scale(input, -128, 127)
            else:
                input_scale = find_scale(input, -128, 127, -1, True)
        input = input.view(-1, input_shape[-1])
        if self._input_per_channel:
            input_scale = input_scale.view(-1, 1)

        out: torch.Tensor = matmul(
            input,
            self.weight.T,
            input_scale,
            self.scale,
            self.bias,
            input_per_channel,
            self._per_channel,
        )
        out = out.view(*input_shape[:-1], -1)
        return out

    @classmethod
    def from_float(
        cls,
        mod: nn.Linear,
        qmin: int = -128,
        qmax: int = 127,
        dtype=torch.int8,
        per_channel: bool = False,
        input_per_channel: bool = False,
    ) -> "DynamicQuantizedLinear":
        """Convert nn.Linear module to dynamically quantied linear layer.

        Args:
            mod (nn.Linear): Linear layer to be quantized
            qmin (int, optional): Quantization minimum range. Defaults to -128.
            qmax (int, optional): Quantization maximum range. Defaults to 127.
            dtype (torch.dtype, optional): Target quantized dtype. Defaults to torch.int8.

        Returns:
            DynamicQuantizedLinear: Dynamically quantized layer
        """
        if per_channel:
            weight, scale = quantize_dynamic_per_channel(mod.weight.detach(), qmin, qmax, dtype, dim=-1)
        else:
            weight, scale = quantize_dynamic_global(mod.weight.detach().contiguous(), qmin, qmax, dtype)
        qlinear = cls(in_features=mod.in_features, out_features=mod.out_features)
        qlinear.weight = nn.Parameter(weight, requires_grad=False)
        qlinear.bias = nn.Parameter(mod.bias, requires_grad=False) if mod.bias is not None else None
        qlinear.scale = torch.tensor(scale.clone().detach()).to("cuda")
        qlinear._per_channel = per_channel
        qlinear._input_per_channel = input_per_channel
        if qlinear.scale.ndim == 0:
            qlinear.scale.unsqueeze_(0)
        return qlinear

    @classmethod
    def from_reference(cls, mod: nn.Module) -> "DynamicQuantizedLinear":
        """Create a dynamic quantized module from a reference quantized
        module

        Args:
            mod (nn.Module): a reference quantized  module, either produced
                by torch.ao.quantization functions or provided by the user

        Returns:
            DynamicQuantizedLinear: Dynamically quantized layer
        """
        qlinear = cls(
            mod.in_features,
            mod.out_features,
        )
        qweight = mod.get_quantized_weight()
        qlinear.bias = mod.bias
        qlinear.weight = nn.Parameter(qweight.int_repr(), requires_grad=False)
        if mod.weight_qscheme == torch.per_tensor_symmetric:
            qlinear.scale = torch.tensor(qweight.q_scale()).to("cuda")
        else:
            scales_fp = []
            for i in range(qweight.shape[0]):
                scale = qweight[i].q_scale()
                scales_fp.append(scale)
            qlinear.scale = torch.tensor(scales_fp, device="cuda")
            qlinear._per_channel = True
        return qlinear
