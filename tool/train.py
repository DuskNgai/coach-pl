import argparse
from pathlib import Path
import sys

sys.path.append(Path.cwd().as_posix())

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from rich import print
import torch

from coach_pl.dataset import BaseDataModule
from coach_pl.module import build_module
from coach_pl.tool.trainer import build_training_trainer, log_time_elasped, setup_cfg
from coach_pl.utils.collect_env import collect_env_info


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A general purpose training script for PyTorch Lightning based projects.")

    parser.add_argument("--config-file", type=Path, metavar="FILE", required=True)
    parser.add_argument("--resume", action="store_true", help="Whether to attempt to resume from the checkpoint directory.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs per node for distributed training.")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None, help="Modify config options at the end of the command. For Yacs configs, use space-separated `PATH.KEY VALUE` pairs.")

    return parser


def main(args: argparse.Namespace) -> None:
    """
    A general purpose training script.
    """

    cfg = setup_cfg(args)

    print(collect_env_info())
    pl.seed_everything(cfg.SEED)
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("config.yaml").open("w") as f:
        f.write(cfg.dump())
    print(cfg.dump())

    trainer, timer = build_training_trainer(args, cfg)
    module = build_module(cfg)
    datamodule = BaseDataModule(cfg)

    if args.resume:
        trainer.fit(module, datamodule=datamodule, ckpt_path="last")
    else:
        trainer.fit(module, datamodule=datamodule)
    log_time_elasped(timer, RunningStage.TRAINING)
    log_time_elasped(timer, RunningStage.VALIDATING)

    trainer.test(module, datamodule=datamodule, ckpt_path="last")
    log_time_elasped(timer, RunningStage.TESTING)
