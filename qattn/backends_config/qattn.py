"""Qattn PyTorch BackendConfig for FX quantization."""

import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)
from torch.ao.quantization.backend_config._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_linear_configs,
    _get_share_qparams_op_configs,
    _get_tensor_info_op_configs,
)

__all__ = [
    "get_qattn_backend_config",
]


def get_qattn_backend_config() -> BackendConfig:
    """Return the `BackendConfig` for the QAttn backend.

    Returns:
        BackendConfig: QAttn Backend Config.
    """
    # dtype configs
    weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    weighted_op_qint8_dynamic_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.float,
        weight_dtype=torch.qint8,
        bias_dtype=torch.qint8,
        is_dynamic=True,
    )
    non_weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
    )

    sdpa_config = (
        BackendPatternConfig(torch.nn.functional.scaled_dot_product_attention)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        .add_dtype_config(non_weighted_op_qint8_dtype_config)
    )
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
        weighted_op_qint8_dynamic_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    tensor_info_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    return (
        BackendConfig("qattn")
        .set_backend_pattern_config(sdpa_config)
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs))
        .set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs))
    )
