import torch
import timm
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from qattn.pt2e.quantizer import (
    QAttnQuantizer,
    get_default_qattn_quantization_config,
)

# initialize quantizer
quantizer = QAttnQuantizer()
quantizer.set_module_type(
    torch.nn.Linear,
    get_default_qattn_quantization_config(
        weight_per_channel=True,
        is_dynamic=True,
        activation_per_channel=False,
    ),
)
# initialize model and sample
model = timm.create_model("vit_large_patch16_224", pretrained=True).to("cuda:0").eval()
sample = torch.randn((1, 3, 224, 224), device="cuda:0")
# export and prepare the model
exported_model = capture_pre_autograd_graph(model, (sample,))
prepared_model = prepare_pt2e(exported_model, quantizer).to(device="cuda:0")

# in PyTorch 2.2 we need to run at least one sample to calculate
# qparams for the weights. TODO: Check for PyTorch 2.3
with torch.inference_mode():
    _ = prepared_model(sample)


converted_model = convert_pt2e(
    prepared_model,
    fold_quantize=True,
)
# invoke lowering via torch compile
model = torch.compile(converted_model, backend="qattn")
_ = model(sample)
