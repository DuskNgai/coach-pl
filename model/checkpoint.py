from pathlib import Path
from typing import Any

from pytorch_lightning.utilities.rank_zero import rank_zero_warn
import torch

TORCH_VERSION: tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])

if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase


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
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def load_pretrained(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """
    Load the pre-trained model from the checkpoint file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        checkpoint_state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        checkpoint_state_dict = ckpt["model"]
    else:
        checkpoint_state_dict = ckpt
    model_state_dict = model.state_dict()

    _strip_prefix_if_present(checkpoint_state_dict, "module.")    # for DistributedDataParallel
    _strip_prefix_if_present(checkpoint_state_dict, "model.")     # for PyTorch Lightning Module
    _strip_prefix_if_present(checkpoint_state_dict, "_orig_mod.") # for torch.compile

    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(
                model_param, torch.nn.parameter.UninitializedParameter
            ):
                continue
            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:

                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    def _get_module_for_key(
                        model: torch.nn.Module, key: str
                    ) -> torch.nn.Module:
                        # foo.bar.param_or_buffer_name -> [foo, bar]
                        key_parts = key.split(".")[:-1]
                        cur_module = model
                        for key_part in key_parts:
                            cur_module = getattr(cur_module, key_part)
                        return cur_module

                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(model, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    msg = model.load_state_dict(checkpoint_state_dict, strict=False)
    rank_zero_warn(f"Loaded pre-trained model from {ckpt_path} with message: {msg}")

    return model
