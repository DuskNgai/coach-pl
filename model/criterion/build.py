from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
import torch.nn

__all__ = ["CRITERION_REGISTRY", "build_criterion"]


CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = "Registry for the model."

def build_criterion(cfg: CfgNode) -> torch.nn.Module:
    """
    Build the criterion defined by `cfg.MODEL.CRITERION.NAME`.
    """
    criterion_name = cfg.MODEL.CRITERION.NAME
    try:
        criterion = CRITERION_REGISTRY.get(criterion_name)(cfg)
    except KeyError as e:
        raise KeyError(CRITERION_REGISTRY) from e

    return criterion
