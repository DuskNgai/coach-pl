from fvcore.common.registry import Registry
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import Dataset

__all__ = [
    "DATASET_REGISTRY",
    "build_dataset",
]

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = "Registry for the dataset."


def build_dataset(cfg: DictConfig, stage: RunningStage) -> Dataset:
    """
    Build the dataset defined by `cfg.DATASET.NAME`.
    """
    dataset_name = cfg.DATASET.NAME
    try:
        dataset = DATASET_REGISTRY.get(dataset_name)(cfg, stage)
    except KeyError as e:
        raise KeyError(DATASET_REGISTRY) from e

    return dataset
