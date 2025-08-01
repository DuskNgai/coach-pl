from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch.nn as nn

__all__ = [
    "CRITERION_REGISTRY",
    "build_criterion",
]

CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = """
Registry for the criterion.
If the name of criterion can be found in `torch.nn`, it will be directly imported.
"""


def build_criterion(cfg: DictConfig) -> nn.Module:
    """
    Build the criterion defined by `cfg.CRITERION.NAME`.
    """
    criterion_name = getattr(cfg, "NAME", cfg.CRITERION.NAME)
    try:
        if hasattr(nn, criterion_name):
            criterion_params = {
                k.lower(): v
                for k, v in cfg.CRITERION.items()
                if k != "NAME"
            }
            criterion = getattr(nn, criterion_name)(**criterion_params)
        else:
            criterion = CRITERION_REGISTRY.get(criterion_name)(cfg)
    except KeyError as e:
        raise KeyError(CRITERION_REGISTRY) from e

    return criterion
