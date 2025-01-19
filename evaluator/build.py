from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch.nn

__all__ = [
    "EVALUATOR_REGISTRY",
    "build_evaluator",
]

EVALUATOR_REGISTRY = Registry("EVALUATOR")
EVALUATOR_REGISTRY.__doc__ = "Registry for the evaluator."


def build_evaluator(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the evaluator defined by `cfg.EVALUATOR.NAME`.
    """
    evaluator_name = cfg.EVALUATOR.NAME
    try:
        evaluator = EVALUATOR_REGISTRY.get(evaluator_name)(cfg)
    except KeyError as e:
        raise KeyError(EVALUATOR_REGISTRY) from e

    return evaluator
