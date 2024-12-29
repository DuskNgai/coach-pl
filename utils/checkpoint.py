import logging
from typing import Any, OrderedDict

from rich.logging import RichHandler
import torch
import torch.nn as nn

logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def strip_prefix_if_present(state_dict: OrderedDict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata # pyre-ignore
    except AttributeError:
        pass
    else:

        # for the metadata dict, the key can be:
        # '': for the DDP module, which we want to remove.
        # 'module': for the actual model.
        # 'module.xx.xx': for the rest.
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def load_pretrained(model: nn.Module, checkpoint_path: str, prefix: str = "") -> None:
    """
    Default load_pretrained function.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        ckpt_state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt_state_dict = ckpt["model"]
    else:
        ckpt_state_dict = ckpt
    strip_prefix_if_present(ckpt_state_dict, prefix)

    msg = model.load_state_dict(ckpt_state_dict, strict=False)
    logger.info(f"Loaded pre-trained model from {checkpoint_path} with message: {msg}")
