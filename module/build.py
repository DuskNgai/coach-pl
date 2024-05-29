from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
import pytorch_lightning as pl

MODULE_REGISTRY = Registry("MODULE")
MODULE_REGISTRY.__doc__ = "Registry for the module, here it is refered to `pl.LightningModule`."

def build_module(cfg: CfgNode) -> pl.LightningModule:
    """
    Build the module defined by `cfg.MODULE.NAME`.
    """
    module_name = cfg.MODULE.NAME
    try:
        module = MODULE_REGISTRY.get(module_name)(cfg)
    except KeyError as e:
        raise KeyError(MODULE_REGISTRY) from e

    return module
