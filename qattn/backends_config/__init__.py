"""QAttn BackendConfig for PyTorch.

BackendConfig deifnes supported ops and modules to be quantized.
For more inforamtion refer to PyTorch documentation
https://github.com/pytorch/pytorch/tree/main/torch/ao/quantization/backend_config
"""

from .qattn import get_qattn_backend_config  # noqa: F401
