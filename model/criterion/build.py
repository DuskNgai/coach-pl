from fvcore.common.registry import Registry
import torch.nn

from configuration import CfgNode

CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = "Registry for the model."

def build_criterion(cfg: CfgNode) -> torch.nn.Module:
    """
    Build the criterion defined by `cfg.MODEL.CRITERION.NAME`.
    """
    criterion_name = cfg.MODEL.CRITERION.NAME
    criterion = CRITERION_REGISTRY.get(criterion_name)(cfg)
    return criterion
