from functools import reduce
from pathlib import Path
from typing import Any

from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from timm.layers import resample_abs_pos_embed
import torch


def _strip_prefix_if_present(state_dict: dict[str, Any], prefix: str) -> None:
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


def load_pretrained(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """
    Load the pre-trained model from the checkpoint file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        ckpt_state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt_state_dict = ckpt["model"]
    else:
        ckpt_state_dict = ckpt
    model_state_dict = model.state_dict()

    _strip_prefix_if_present(ckpt_state_dict, "module.")    # for DistributedDataParallel
    _strip_prefix_if_present(ckpt_state_dict, "model.")     # for PyTorch Lightning Module
    _strip_prefix_if_present(ckpt_state_dict, "_orig_mod.") # for torch.compile

    for name, ckpt_param in ckpt_state_dict.items():
        # Extra parameters in the checkpoint that are not in the model
        model_param = model_state_dict.get(name)
        if model_param is None:
            continue

        shape_ckpt_param = tuple(ckpt_param.shape)
        shape_model_param = tuple(model_param.shape)

        # Parameters that fit exactly
        if shape_model_param == shape_ckpt_param:
            continue

        # Parameters that need to be reshaped
        if "pos_embed" in name:
            num_embed_dim = shape_ckpt_param[-1]
            num_model_patches = model.patch_embed.num_patches
            num_ckpt_patches = reduce(lambda x, y: x * y, shape_ckpt_param[1 :-1])
            num_prefix_tokens = 1 if "cls_token" in ckpt_state_dict else 0

            if num_model_patches + num_prefix_tokens == num_ckpt_patches:
                new_pos_embed = ckpt_param.reshape(-1, num_ckpt_patches, num_embed_dim)
                new_pos_embed = new_pos_embed[:, num_prefix_tokens :, :].view(shape_model_param)
            else:
                new_pos_embed = resample_abs_pos_embed(
                    ckpt_param.reshape(-1, num_ckpt_patches, num_embed_dim)[:, num_prefix_tokens :, :],
                    new_size=model.patch_embed.grid_size,
                    num_prefix_tokens=0,
                ).view(shape_model_param)

            ckpt_state_dict[name] = new_pos_embed
        else:
            ckpt_state_dict.pop(name)

    msg = model.load_state_dict(ckpt_state_dict, strict=False)
    rank_zero_warn(f"Loaded pre-trained model from {ckpt_path} with message: {msg}")

    return model
