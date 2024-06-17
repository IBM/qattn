"""Graph manipulation transforms for simplifying model graph."""

from typing import Any, Dict, Tuple, Optional, Union

import torch.fx as fx
from timm.layers.drop import DropPath
from torch import nn
from torch.fx.node import Argument, Target


__all__ = [
    "remove_identity_layers",
]


class LayerRemover(fx.Transformer):
    """Transformer to remove identity layers during inference.

    Args:
        module (nn.Module): Captured model graph.
        types (Optional[Set[nn.Module]], optional): types of layers to remove. Defaults to
            {nn.Identity, DropPath, nn.Dropout}.

    Attributes:
        DEFAULT_TYPES (set): default layers to be removed from the model's graph.
    """

    DEFAULT_TYPES = (
        nn.Identity,
        DropPath,
        nn.Dropout,
    )

    def __init__(self, module: nn.Module, types: Optional[Tuple[nn.Module]] = None):
        super().__init__(module)
        self.types = types or self.DEFAULT_TYPES

    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """Transform the graph if the module is in selected types.

        Args:
            target (Target): target module.
            args (Tuple[Argument, ...]): Args for the module.
            kwargs (Dict[str, Any]): Kwargs to the module

        Returns:
            Any: return the node as is or args[0] to removed module.
        """
        if isinstance(self.submodules[target], self.types):
            assert len(args) == 1
            return args[0]
        else:
            return super().call_module(target, args, kwargs)


def remove_identity_layers(model: Union[fx.GraphModule, nn.Module]) -> fx.GraphModule:
    """Removes idenity layers in model graph.

    Removes operation like dropout, DropPath, Identity in the graph.

    Args:
        model (Union[fx.GraphModule, nn.Module]): model for transformation.

    Returns:
        fx.GraphModule: Transformed model without identity layers.
    """
    fx_model = fx.symbolic_trace(model)

    return LayerRemover(fx_model).transform()
