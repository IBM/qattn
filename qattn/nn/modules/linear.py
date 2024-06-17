"""Statically quantized linear layer"""
from typing import Union
import torch
import torch.nn as nn

from qattn.nn.functional import matmul
from qattn.quantize import quantize_global


class QuantizedLinear(nn.Linear):
    """Statically quantized linear layer.

    Args:
        scale (torch.Tensor): Quantized weight scale factor.
        out_scale (torch.Tensor): Output scale factor.
        in_features (int): size of each input sample
        out_features(int): size of each output sample
        bias (bool): If set to `False`, the layer will not learn an additive bias. Default: `True`
    """

    scale: torch.Tensor
    out_scale: torch.Tensor
    _per_channel: bool = False
    _input_per_channel: bool = False

    def forward(
        self,
        input: torch.Tensor,
        input_scale: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """Defines the forward computation of linear layer.

        Args:
            input (torch.Tensor): input torch tensor.
            input_scale (Union[torch.Tensor, float]): input quantization scale facotr.

        Returns:
            torch.Tensor: output tensor.
        """
        if input.dtype in {torch.half, torch.float32}:
            input = quantize_global(input, input_scale.view(-1, 1), torch.int8)
        input_shape = input.shape
        input = input.view(-1, input_shape[-1])
        out_scale = self.out_scale
        if self._input_per_channel:
            if input_scale.dim() == 1:
                input_scale = input_scale.repeat(input_shape[0])
            out_scale = out_scale.repeat(input_scale.shape[0])

        out: torch.Tensor = matmul(
            input,
            self.weight.T,
            input_scale,
            self.scale,
            out_scale,
            self.bias,
            self._input_per_channel,
            self._per_channel,
        )
        out = out.view(*input_shape[:-1], -1)
        return out

    @classmethod
    def from_reference(cls, mod: nn.Module, out_scale: Union[torch.Tensor, float]) -> "QuantizedLinear":
        """Create a static quantized module from a reference quantized module.

        Args:
            mod (nn.Module): a reference quantized  module, either produced
                by torch.ao.quantization functions or provided by the user
            out_scale (Union[torch.Tensor, float]): an out scale quantization factor.

        Returns:
            QuantizedLinear: Statically quantized layer
        """
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
        )
        qweight = mod.get_quantized_weight()
        if mod.bias is not None:
            qlinear.bias = nn.Parameter(mod.bias.to(device="cuda"), requires_grad=False)
        qlinear.weight = nn.Parameter(qweight.int_repr().to("cuda"), requires_grad=False)
        qlinear.scale = torch.tensor(mod.weight_scale).to(device="cuda")
        qlinear.out_scale = torch.tensor(out_scale).to(device="cuda")
        qlinear._per_channel = qlinear.scale.numel() > 1
        qlinear._input_per_channel = qlinear.out_scale.numel() > 1
        if qlinear.out_scale.ndim == 0:
            qlinear.out_scale.unsqueeze_(0)
        return qlinear
