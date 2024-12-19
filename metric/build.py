from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch.nn

__all__ = [
    "METRIC_REGISTRY",
    "build_metric",
]

METRIC_REGISTRY = Registry("METRIC")
METRIC_REGISTRY.__doc__ = "Registry for the metric."


def build_metric(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the metric defined by `cfg.METRIC.NAME`.
    """
    metric_name = cfg.METRIC.NAME
    try:
        metric = METRIC_REGISTRY.get(metric_name)(cfg)
    except KeyError as e:
        raise KeyError(METRIC_REGISTRY) from e

    return metric
