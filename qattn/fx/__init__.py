"""QAttn FX Graph Quantization."""

import torch.fx as fx
from torch.ao.quantization import quantize_fx
from .qconfig import get_default_qconfig  # noqa: F401
from qattn.backends_config import get_qattn_backend_config
from qattn.fx import lower


__all__ = ["convert"]


def convert(model: fx.GraphModule, quant_sdpa: bool = True, is_dynamic: bool = False) -> fx.GraphModule:
    """Convert the capture model to quantized model with QAttn kernels.

    Args:
        model (fx.GraphModule): Calibrated the model
        quant_sdpa (bool, optional): Quantize attention. Defaults to True.
        is_dynamic (bool, optional): is dynamic quantization. Defaults to False.

    Returns:
        fx.GraphModule: Quantized model.
    """
    model = quantize_fx.convert_to_reference_fx(model, backend_config=get_qattn_backend_config())
    return lower.lower(model, is_dynamic=is_dynamic, quant_sdpa=quant_sdpa)
