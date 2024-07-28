import argparse
from pathlib import Path
import sys

sys.path.append(Path.cwd().as_posix())

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.states import RunningStage
from rich import print
import torch

from coach_pl.configuration import CfgNode
from coach_pl.dataset import BaseDataModule
from coach_pl.module import build_module
from coach_pl.tool.trainer import build_testing_trainer, log_time_elasped, setup_cfg
from coach_pl.utils.collect_env import collect_env_info


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A general purpose testing script for PyTorch Lightning based projects.")

    parser.add_argument("--config-file", type=Path, metavar="FILE", required=True)
    parser.add_argument("--ckpt-path", type=Path, metavar="FILE", required=True, help="Path to the checkpoint file.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None, help="Modify config options at the end of the command. For Yacs configs, use space-separated `PATH.KEY VALUE` pairs.")

    return parser


def main(args: argparse.Namespace) -> None:
    """
    A general purpose testing script.
    """

    cfg = setup_cfg(args)

    seed_everything(cfg.SEED, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")

    trainer, timer = build_testing_trainer(cfg)
    module = build_module(cfg)
    datamodule = BaseDataModule(cfg)

    if trainer.global_rank == 0:
        print(collect_env_info())
        print(CfgNode.to_yaml(cfg))

    trainer.test(module, datamodule=datamodule, ckpt_path=args.ckpt_path)
    log_time_elasped(timer, RunningStage.TESTING)
