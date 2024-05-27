import torch.nn as nn

from configuration import CfgNode

from .build import CRITERION_REGISTRY

@CRITERION_REGISTRY.register()
def build_pytorch_criterion(cfg: CfgNode) -> nn.Module:
    """
    Build the builtin criterion defined by `cfg.MODEL.CRITERION.NAME`.
    """
    loss_name = cfg.MODEL.CRITERION.NAME

    try:
        criterion = getattr(nn, loss_name)(**cfg.MODEL.CRITERION.PARAMS)
    except AttributeError:
        raise AttributeError(f"Criterion {loss_name} not found in torch.nn")

    return criterion
