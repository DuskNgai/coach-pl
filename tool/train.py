import argparse
from pathlib import Path
import sys

sys.path.append(Path.cwd().as_posix())

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.states import RunningStage
import torch

from coach_pl.dataset import BaseDataModule
from coach_pl.module import build_module

from .trainer import (
    build_training_trainer,
    log_configurations,
    log_time_elasped,
    setup_cfg,
)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A general purpose training script for PyTorch Lightning based projects."
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        metavar="FILE",
        required=True,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="FILE",
        help="The path to the checkpoint to resume from.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per node for distributed training.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help=
        "Modify config options at the end of the command. For Yacs configs, use space-separated `PATH.KEY VALUE` pairs.",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """
    A general purpose training script.
    """

    cfg = setup_cfg(args)
    log_configurations(cfg)

    torch.set_float32_matmul_precision("high")

    trainer, timer = build_training_trainer(args, cfg)
    seed_everything(cfg.SEED + trainer.global_rank, workers=True)

    module = build_module(cfg)
    datamodule = BaseDataModule(cfg)

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)
    log_time_elasped(timer, RunningStage.TRAINING)
    log_time_elasped(timer, RunningStage.VALIDATING)

    if cfg.TRAINER.PROFILER is None:
        trainer.test(module, datamodule=datamodule, ckpt_path="last")
        log_time_elasped(timer, RunningStage.TESTING)
