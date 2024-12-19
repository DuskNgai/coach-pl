from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch.nn

__all__ = [
    "CRITERION_REGISTRY",
    "build_criterion",
]

CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = "Registry for the criterion."


def build_criterion(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the criterion defined by `cfg.CRITERION.NAME`.
    """
    criterion_name = cfg.CRITERION.NAME
    try:
        criterion = CRITERION_REGISTRY.get(criterion_name)(cfg)
    except KeyError as e:
        raise KeyError(CRITERION_REGISTRY) from e

    return criterion
