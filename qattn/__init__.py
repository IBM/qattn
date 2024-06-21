from .fx import get_default_qconfig  # noqa: F401
from .fx import convert  # noqa: F401
from . import nn  # noqa: F401
from .pt2e.dynamo.backend import backends  # noqa: F401

__version__ = "0.1.1"


def version() -> str:
    """Version of the qattn package.

    Returns:
        str: version string.
    """
    return __version__
