import copy
from typing import Dict, List, Callable

import torch
import torch.fx as fx
import torch.nn.functional as F

from torch.ao.quantization.observer import (
    PlaceholderObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver,
)
from torch.ao.quantization.quantizer import xnnpack_quantizer as xnq
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationSpec,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
    OperatorConfig,
    OperatorPatternType,
)

from qattn.pt2e.annotators import _annotate_sdpa, _annotate_linear  # noqa: F401
from qattn.observer import MovingAveragePerLastChannelMinMaxObserver


def get_default_qattn_quantization_config(
    weight_per_channel: bool = False,
    activation_per_channel: bool = False,
    is_dynamic: bool = False,
) -> QuantizationConfig:
    """Get default QAttn quantization config.

    Default returns static symmetric quantization.

    Args:
        weight_per_channel (bool, optional): Determines if the weights should be quantized per channel.
            Defaults to False.
        activation_per_channel (bool, optional): Determines if the activation input and output should be quantized per
          channel. Defaults to False.
        is_dynamic (bool, optional): Sets the quantization to be dynamic quantization. Defaults to False.

    Returns:
        QuantizationConfig: Quantization configuration.
    """
    if is_dynamic:
        act_obs = PlaceholderObserver.with_args(qscheme=torch.per_tensor_symmetric)
    else:
        if activation_per_channel:
            act_obs = MovingAveragePerLastChannelMinMaxObserver
        else:
            act_obs = MovingAverageMinMaxObserver
    if activation_per_channel:
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            is_dynamic=is_dynamic,
            observer_or_fake_quant_ctr=act_obs,
            ch_axis=1 if activation_per_channel else None,
        )

    else:
        act_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=is_dynamic,
            observer_or_fake_quant_ctr=act_obs,
        )

    weight_quantization_spec = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_channel_symmetric if weight_per_channel else torch.per_tensor_symmetric,
        is_dynamic=False,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver if weight_per_channel else MinMaxObserver,
    )

    bias_quantization_spec = QuantizationSpec(
        torch.float32,
        observer_or_fake_quant_ctr=PlaceholderObserver,
    )

    quantization_spec = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec if not is_dynamic else None,
        weight_quantization_spec,
        bias_quantization_spec,
    )
    return quantization_spec


def _get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
        get_default_qattn_quantization_config(),
    ]:
        ops = _supported_symmetric_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(OperatorConfig(quantization_config, pattern_list))
    return copy.deepcopy(supported_config_and_operators)


def _supported_symmetric_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        "linear": [[torch.nn.Linear], [F.linear]],
    }
    return copy.deepcopy(supported_operators)


def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_symmetric_config_and_operators()


class QAttnQuantizer(xnq.XNNPACKQuantizer):
    supported_config_and_operators = _get_supported_config_and_operators()
    STATIC_QAT_ONLY_OPS = []
    STATIC_OPS = [
        "linear",
        "scaled_dot_product_attention",
    ]
    DYNAMIC_OPS = ["linear"]

    def _annotate_for_static_quantization_config(self, model: fx.GraphModule) -> fx.GraphModule:
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(model, config, xnq._get_module_name_filter(module_name))

        for module_type, config in self.module_type_config.items():
            self._annotate_all_static_patterns(model, config, xnq._get_module_type_filter(module_type))

        # annotate attention
        module_type = torch.nn.functional.scaled_dot_product_attention
        if self.module_type_config.get(module_type, False):
            self._annotate_all_static_patterns(model, config, _get_sdpa_fn_type_filter(module_type))

        return model

    def _annotate_for_dynamic_quantization_config(self, model: fx.GraphModule) -> fx.GraphModule:
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_dynamic_patterns(model, config, xnq._get_module_name_filter(module_name))

        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_all_dynamic_patterns(model, config, xnq._get_module_type_filter(module_type))

        self._annotate_all_dynamic_patterns(
            model,
            self.global_config,
            xnq._get_not_module_type_or_name_filter(tp_list, module_name_list),
        )

        return model

    def annotate(self, model: fx.GraphModule) -> fx.GraphModule:
        """Annotate the model with quantization spec

        Args:
            model (fx.GraphModule): captured model

        Returns:
            fx.GraphModule: Annotated model with quantization spec
        """
        if self.global_config and self.global_config.input_activation.is_dynamic:  # type: ignore[union-attr]
            model = self._annotate_for_dynamic_quantization_config(model)
        elif (
            torch.nn.Linear in self.module_type_config
            and self.module_type_config[torch.nn.Linear].input_activation.is_dynamic
        ):
            model = self._annotate_for_dynamic_quantization_config(model)
        else:
            model = self._annotate_for_static_quantization_config(model)
        return model


def _get_sdpa_fn_type_filter(tp: Callable):
    """Get the sdpa_fn_type_filter function for a scaled dot product attention fn type, the filter accepts
    a node and checks if the node comes from a function that is scaled dot product attention.

    For example:
        node: scaled_dot_product_attention = call_function[...](...)
    """

    def sdpa_type_filter(n: fx.Node) -> bool:
        nn_module_stack = n.meta.get("source_fn_stack", {})
        types = [t for _, t in nn_module_stack]
        return tp in types

    return sdpa_type_filter
