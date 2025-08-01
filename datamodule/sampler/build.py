from fvcore.common.registry import Registry
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import (
    Dataset,
    Sampler,
)

__all__ = [
    "SAMPLER_REGISTRY",
    "build_sampler",
]

SAMPLER_REGISTRY = Registry("SAMPLER")
SAMPLER_REGISTRY.__doc__ = "Registry for dataset the sampler."


def build_sampler(cfg: DictConfig, stage: RunningStage, dataset: Dataset) -> Sampler:
    """
    Build the sampler defined by `cfg.DATAMODULE.SAMPLER.NAME`.
    """
    sampler_name = getattr(cfg, "NAME", cfg.DATAMODULE.SAMPLER.NAME)
    try:
        sampler = SAMPLER_REGISTRY.get(sampler_name)(cfg, stage, dataset)
    except KeyError as e:
        raise KeyError(SAMPLER_REGISTRY) from e

    return sampler
