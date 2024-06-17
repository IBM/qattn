from torch.ao.quantization.backend_config import BackendConfig

from qattn.backends_config import get_qattn_backend_config


def test_get_backend_config():
    cfg = get_qattn_backend_config()
    assert isinstance(cfg, BackendConfig), "Generated config is of BackendConfig type"
