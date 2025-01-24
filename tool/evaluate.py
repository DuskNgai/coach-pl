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
    build_evaluating_trainer,
    log_configurations,
    log_time_elasped,
    setup_cfg,
)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A general purpose testing script for PyTorch Lightning based projects."
    )

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
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help=
        "Modify config options at the end of the command. For Yacs configs, use space-separated `PATH.KEY VALUE` pairs.",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """
    A general purpose evaluation script.
    """

    cfg = setup_cfg(args)
    log_configurations(cfg)

    torch.set_float32_matmul_precision("high")

    trainer, timer = build_evaluating_trainer(cfg)
    seed_everything(cfg.SEED + trainer.global_rank, workers=True)

    module = build_module(cfg)
    datamodule = BaseDataModule(cfg)

    trainer.test(module, datamodule=datamodule, ckpt_path=args.ckpt_path)
    log_time_elasped(timer, RunningStage.TESTING)
