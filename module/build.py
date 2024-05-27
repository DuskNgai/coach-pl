import pytorch_lightning as pl

from configuration import CfgNode
from fvcore.common.registry import Registry

MODULE_REGISTRY = Registry("MODULE")
MODULE_REGISTRY.__doc__ = "Registry for the module, here it is refered to `pl.LightningModule`."

def build_module(cfg: CfgNode) -> pl.LightningModule:
    """
    Build the module defined by `cfg.MODULE.NAME`.
    """
    module_name = cfg.MODULE.NAME
    module = MODULE_REGISTRY.get(module_name)(cfg)
    return module
