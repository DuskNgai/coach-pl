from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
import torch

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = "Registry for the model."

def build_model(cfg: CfgNode) -> torch.nn.Module:
    """
    Build the model defined by `cfg.MODEL.NAME`.
    It moves the model to the device defined by `cfg.MODEL.DEVICE`.
    It does not load checkpoints from `cfg`.
    """
    model_name = cfg.MODEL.NAME
    try:
        model = MODEL_REGISTRY.get(model_name)(cfg)
    except KeyError as e:
        raise KeyError(MODEL_REGISTRY) from e

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
