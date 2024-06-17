import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import quantize_fx, QConfigMapping
import timm
import qattn
from qattn.backends_config.qattn import get_qattn_backend_config

# create model
model = timm.create_model("vit_large_patch16_224", pretrained=True)
model = model.eval()

# get default qconfig
qconfig = qattn.get_default_qconfig()
# define QConfigMapping
mapping = (
    QConfigMapping()
    .set_global(None)
    .set_object_type(nn.Linear, qconfig)
    .set_object_type(F.scaled_dot_product_attention, qconfig)
)

# example input, can be arbitrary.
example_input = torch.randn(1, 3, 224, 224)

model = quantize_fx.prepare_fx(
    model,
    example_inputs=example_input,
    qconfig_mapping=mapping,
    backend_config=get_qattn_backend_config(),
).to("cuda")

# calibrate the model
for _ in range(5):
    with torch.inference_mode():
        x = torch.randn(1, 3, 224, 224).to("cuda")
        _ = model(x)

# convert the model
model = qattn.convert(model)

model(example_input.to("cuda"))

print(model)
