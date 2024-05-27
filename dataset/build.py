import torch.utils.data

from pytorch_lightning.trainer.states import RunningStage

from configuration import CfgNode
from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = "Registry for the dataset."

def build_dataset(cfg: CfgNode, stage: RunningStage) -> torch.utils.data.Dataset:
    """
    Build the dataset defined by `cfg.DATASET.NAME`.
    """
    dataset_name = cfg.DATASET.NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg, stage)
    return dataset
