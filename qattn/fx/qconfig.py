"""Default quantization configs for QAttn backend."""

import torch
from torch.ao.quantization import (
    MinMaxObserver,
    PlaceholderObserver,
    PerChannelMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.qconfig import QConfig

# default static INT8 qconfig
default_qconfig = QConfig(
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    activation=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
)

default_per_channel_qconfig = QConfig(
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    activation=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
)

default_per_channel_input_qconfig = QConfig(
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    activation=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=1
    ),
)


# default dynamic INT8 qconfig
default_dynamic_qconfig = QConfig(
    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    activation=PlaceholderObserver.with_args(dtype=torch.qint8, is_dynamic=True),
)

default_per_channel_dynamic_qconfig = QConfig(
    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    activation=PlaceholderObserver.with_args(dtype=torch.qint8, is_dynamic=True),
)


def get_default_qconfig(
    is_dynamic: bool = False, input_per_channel: bool = False, per_channel: bool = False
) -> QConfig:
    """Returns default qconfig for QAttn backend.

    Static quantization default qconfig:

        default_qconfig = QConfig(
           weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
           activation=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        )

    Dynamic quantization default qconfig:

        default_dynamic_qconfig = QConfig(
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
            activation=PlaceholderObserver.with_args(dtype=torch.qint8, is_dynamic=True),
        )

    Args:
        is_dynamic (bool, optional): If true returns qconfig for dynamic quantization. Defaults to False.

    Returns:
        QConfig: Default QConfig configuration.
    """
    if is_dynamic:
        if per_channel:
            return default_per_channel_dynamic_qconfig
        return default_dynamic_qconfig
    if per_channel:
        if input_per_channel:
            return default_per_channel_input_qconfig
        return default_per_channel_qconfig
    return default_qconfig
