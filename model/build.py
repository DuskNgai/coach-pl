import torch

from configuration import CfgNode
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = "Registry for the model."

def build_model(cfg: CfgNode) -> torch.nn.Module:
    """
    Build the model defined by `cfg.MODEL.NAME`.
    It does not load checkpoints from `cfg`.
    """
    model_name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(model_name)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
