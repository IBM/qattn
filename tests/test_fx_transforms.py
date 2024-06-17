import torch.nn as nn
from qattn.fx import transforms as FT


def test_remove_identity_dropout():
    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(32, 32)
            self.identity = nn.Identity()
            self.dropout = nn.Dropout()

        def forward(self, x):
            return self.dropout(self.identity(self.linear(x)))

    model = M()
    transformed_model = FT.remove_identity_layers(model)

    for module in transformed_model.modules():
        assert not isinstance(module, (nn.Dropout, nn.Identity))

