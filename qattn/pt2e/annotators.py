import torch
from typing import Callable, Optional, List

import torch.fx as fx
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    register_annotator,
    QuantizationConfig,
    get_weight_qspec,
    get_bias_qspec,
    _is_annotated,
    _mark_nodes_as_annotated,
)
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.utils import _annotate_input_qspec_map, _annotate_output_qspec


def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    """Get input activation quantization specification.

    Args:
        quantization_config (Optional[QuantizationConfig]): Quantization configuration for the node.

    Returns:
        Optional[QuantizationSpec]: Activation input quantization spec.
    """
    if quantization_config is None:
        return None
    if quantization_config.input_activation is None:
        return None
    quantization_spec = quantization_config.input_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]
    return quantization_spec


def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    """Get output activation quantziation spec.

    Args:
        quantization_config (Optional[QuantizationConfig]): Quantization configuration for the node.

    Returns:
        Optional[QuantizationSpec]: Activation output quantization spec.
    """
    if quantization_config is None:
        return None
    if quantization_config.output_activation is None:
        return None
    quantization_spec = quantization_config.output_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]
    return quantization_spec


@register_annotator("scaled_dot_product_attention")
def _annotate_sdpa(
    gm: fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[fx.Node], bool]] = None,
) -> Optional[List[List[fx.Node]]]:
    """Annotate scaled dot product attention node.

    Args:
        gm (fx.GraphModule): Captured graph module.
        quantization_config (Optional[QuantizationConfig]): Quantization configuration.
        filter_fn (Optional[Callable[[fx.Node], bool]], optional): Filter function. Defaults to None.

    Returns:
        Optional[List[List[fx.Node]]]: annotated nodes list
    """
    annotated_partitions = []

    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.scaled_dot_product_attention.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        if len(node.args) == 3:
            bias = None
        elif len(node.args) == 4:
            bias = node.args[-1]
        if not _is_annotated([node]):
            if input_act_qspec.is_dynamic:
                continue
            nodes_to_mark_annotated = [node]
            if bias is not None:
                _annotate_input_qspec_map(node, bias, bias_qspec)
                nodes_to_mark_annotated.append(bias)
            _annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions


@register_annotator("linear")
def _annotate_linear(
    gm: fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[fx.Node], bool]] = None,
) -> Optional[List[List[fx.Node]]]:
    """Annotate the linear operation.

    Args:
        gm (fx.GraphModule): Captured graph module.
        quantization_config (Optional[QuantizationConfig]): Quantization configuration.
        filter_fn (Optional[Callable[[fx.Node], bool]], optional): Filter function. Defaults to None.

    Returns:
        Optional[List[List[fx.Node]]]: annotated nodes list
    """
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        act_node = node.args[0]
        weight_node = node.args[1]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[2]

        if _is_annotated([node]) is False:  # type: ignore[list-item]
            _annotate_input_qspec_map(
                node,
                act_node,
                input_act_qspec,
            )
            _annotate_input_qspec_map(
                node,
                weight_node,
                weight_qspec,
            )
            nodes_to_mark_annotated = [node, weight_node]
            if bias_node:
                _annotate_input_qspec_map(
                    node,
                    bias_node,
                    bias_qspec,
                )
                nodes_to_mark_annotated.append(bias_node)
            _annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions
