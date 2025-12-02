import argparse
import functools
from pathlib import Path
import sys

sys.path.append(Path.cwd().as_posix())

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import torch

from coach_pl.datamodule import BaseDataModule
from coach_pl.module import build_module

from .trainer import (
    build_evaluating_trainer,
    log_configurations,
    log_time_elasped,
    setup_cfg,
)

# HACK: For torch.load(..., weights_only=False).

_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _trusted_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A general purpose testing script for PyTorch Lightning based projects.")

    parser.add_argument(
        "--config-file",
        type=Path,
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        metavar="FILE",
        required=True,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per node for distributed evaluation.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed evaluation.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help="Modify config options at the end of the command. For Yacs configs, use space-separated `PATH.KEY VALUE` pairs.",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """
    A general purpose evaluation script.
    """

    cfg = setup_cfg(args)
    log_configurations(cfg)

    torch.set_float32_matmul_precision("high")

    trainer, timer = build_evaluating_trainer(args, cfg)
    seed_everything(cfg.SEED + trainer.global_rank, workers=True)

    module = build_module(cfg)
    datamodule = BaseDataModule(cfg)

    if args.ckpt_path.is_dir():
        # `pytorch_model.bin` is the default name for checkpoint converted by DeepSpeed.
        ckpt_name = args.ckpt_path.joinpath("pytorch_model.bin")
        if not ckpt_name.exists():
            convert_zero_checkpoint_to_fp32_state_dict(args.ckpt_path, ckpt_name)
        trainer.test(module, datamodule=datamodule, ckpt_path=ckpt_name)
    else:
        trainer.test(module, datamodule=datamodule, ckpt_path=args.ckpt_path)
    log_time_elasped(timer, RunningStage.TESTING)
