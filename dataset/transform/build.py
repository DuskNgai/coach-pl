from fvcore.common.registry import Registry
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torchvision.transforms import v2

__all__ = [
    "TRANSFORM_REGISTRY",
    "build_transform",
]

TRANSFORM_REGISTRY = Registry("TRANSFORM")
TRANSFORM_REGISTRY.__doc__ = "Registry for the data transform."


def build_transform(cfg: DictConfig, stage: RunningStage) -> v2.Compose:
    """
    Build the dataset defined by `cfg.DATASET.TRANSFORM.NAME`.
    """
    transform_name = cfg.DATASET.TRANSFORM.NAME
    try:
        transform = TRANSFORM_REGISTRY.get(transform_name)(cfg, stage)
    except KeyError as e:
        raise KeyError(TRANSFORM_REGISTRY) from e

    return transform
