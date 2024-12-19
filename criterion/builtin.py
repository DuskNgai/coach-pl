from omegaconf import DictConfig
import torch.nn as nn

from .build import CRITERION_REGISTRY

__all__ = ["build_pytorch_criterion"]


@CRITERION_REGISTRY.register()
def build_pytorch_criterion(cfg: DictConfig) -> nn.Module:
    """
    Build the builtin criterion defined by `cfg.CRITERION.TYPE`.
    """
    loss_type = cfg.CRITERION.TYPE

    try:
        criterion = getattr(nn, loss_type)()
    except AttributeError:
        raise AttributeError(f"Criterion {loss_type} not found in torch.nn")

    return criterion
