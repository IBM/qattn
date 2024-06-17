import torch
from torch.ao.quantization.observer import MovingAveragePerChannelMinMaxObserver


class MovingAveragePerLastChannelMinMaxObserver(MovingAveragePerChannelMinMaxObserver):
    """Wrapper around MovingAveragePerChannelMinMaxObserver.

    Helper class to average per last channel always.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(
            averaging_constant,
            ch_axis,
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            eps,
            is_dynamic,
            **kwargs,
        )
        self.ch_axis = 0
        self.first_run = True

    def forward(self, x_orig):
        if self.first_run:
            self.ch_axis = x_orig.ndim - 2
            self.first_run = False
        return super().forward(x_orig)
