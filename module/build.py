from fvcore.common.registry import Registry
from omegaconf import DictConfig
import pytorch_lightning as pl

__all__ = [
    "MODULE_REGISTRY",
    "build_module",
]

MODULE_REGISTRY = Registry("MODULE")
MODULE_REGISTRY.__doc__ = "Registry for the module, here it is refered to `pl.LightningModule`."


def build_module(cfg: DictConfig) -> pl.LightningModule:
    """
    Build the module defined by `cfg.MODULE.NAME`.
    """
    module_name = cfg.MODULE.NAME
    try:
        module = MODULE_REGISTRY.get(module_name)(cfg)
    except KeyError as e:
        raise KeyError(MODULE_REGISTRY) from e

    return module
