from fvcore.common.config import CfgNode
import torch.nn as nn

from .build import CRITERION_REGISTRY

@CRITERION_REGISTRY.register()
def build_pytorch_criterion(cfg: CfgNode) -> nn.Module:
    """
    Build the builtin criterion defined by `cfg.MODEL.CRITERION.TYPE`.
    """
    loss_type = cfg.MODEL.CRITERION.TYPE

    try:
        criterion = getattr(nn, loss_type)()
    except AttributeError:
        raise AttributeError(f"Criterion {loss_type} not found in torch.nn")

    return criterion
