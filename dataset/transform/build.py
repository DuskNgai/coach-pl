from fvcore.common.registry import Registry
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
import torch.utils.data

__all__ = ["TRANSFORM_REGISTRY", "build_transform"]


TRANSFORM_REGISTRY = Registry("TRANSFORM")
TRANSFORM_REGISTRY.__doc__ = "Registry for the transform."

def build_transform(cfg: DictConfig, stage: RunningStage) -> torch.utils.data.Dataset:
    """
    Build the dataset defined by `cfg.DATASET.TRANSFORM.NAME`.
    """
    transform_name = cfg.DATASET.TRANSFORM.NAME
    try:
        transform = TRANSFORM_REGISTRY.get(transform_name)(cfg, stage)
    except KeyError as e:
        raise KeyError(TRANSFORM_REGISTRY) from e

    return transform
