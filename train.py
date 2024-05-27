import argparse
import datetime
from pathlib import Path

from rich import print
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.trainer.states import RunningStage
import torch

from configuration import CfgNode
from module import build_module
from dataset import BaseDataModule
from utils.collect_env import collect_env_info


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=Path, metavar="FILE", required=True)
    parser.add_argument("--resume", action="store_true", help="Whether to attempt to resume from the checkpoint directory.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs per node for distributed training.")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None, help="Modify config options at the end of the command. For Yacs configs, use space-separated `PATH.KEY VALUE` pairs.")

    args = parser.parse_args()
    return args


def setup_cfg(args: argparse.Namespace) -> CfgNode:
    """
    Create configs from default settings, file, and command-line arguments.
    """
    cfg = CfgNode(CfgNode.load_yaml_with_base(args.config_file))
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def log_time_elasped(timer: Timer, stage: RunningStage) -> None:
    elasped_time = datetime.timedelta(seconds=timer.time_elapsed(stage))
    print(f"Running time for {stage}: {elasped_time}")


def main(args):
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

    timer = Timer()
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="ddp" if args.num_gpus > 1 else "auto",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        precision="16-mixed" if cfg.TRAINER.MIXED_PRECISION else "32-true",
        logger=[
            CSVLogger(output_dir, "csv_log"),
            TensorBoardLogger(output_dir, "tb_log"),
        ],
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir.joinpath("best_ckpts"),
                filename="{epoch}-{" + cfg.TRAINER.MONITOR_METRIC + ":.2f}",
                monitor=cfg.TRAINER.MONITOR_METRIC,
                mode="min" if "loss" in cfg.TRAINER.MONITOR_METRIC else "max",
                save_top_k=3,
                save_last=True,
                every_n_epochs=1,
            ),
            ModelCheckpoint(
                dirpath=output_dir.joinpath("regular_ckpts"),
                filename="{epoch}-{" + cfg.TRAINER.MONITOR_METRIC + ":.2f}",
                monitor="epoch",
                mode="max",
                save_top_k=10,
                every_n_epochs=5,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            timer
        ],
        max_epochs=cfg.TRAINER.MAX_EPOCHS,
        gradient_clip_val=cfg.TRAINER.CLIP_GRAD.VALUE if cfg.TRAINER.CLIP_GRAD.ENABLED else None,
        gradient_clip_algorithm=cfg.TRAINER.CLIP_GRAD.ALGORITHM if cfg.TRAINER.CLIP_GRAD.ENABLED else None,
        deterministic=cfg.TRAINER.DETERMINISTIC,
        benchmark=cfg.TRAINER.BENCHMARK,
        log_every_n_steps=cfg.TRAINER.LOG_EVERY_N_STEPS,
        detect_anomaly=cfg.TRAINER.DETECT_ANOMALY,
    )

    module = build_module(cfg)
    datamodule = BaseDataModule(cfg)

    if args.resume:
        trainer.fit(module, datamodule=datamodule, ckpt_path="last")
    else:
        trainer.fit(module, datamodule=datamodule)
    log_time_elasped(timer, RunningStage.TRAINING)
    log_time_elasped(timer, RunningStage.VALIDATING)

    trainer.test(module, datamodule=datamodule, ckpt_path="best")
    log_time_elasped(timer, RunningStage.TESTING)


if __name__ == "__main__":
    args = parse_args()
    main(args)
