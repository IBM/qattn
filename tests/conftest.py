import pytest
import torch.nn as nn
from timm.models.vision_transformer import Attention


@pytest.fixture
def mlp():
    mlp = nn.Sequential(*[nn.Linear(16, 16, bias=True), nn.ReLU(), nn.Linear(16, 16, bias=True)])
    return mlp


@pytest.fixture
def attention_block():
    return Attention(192, 3)
