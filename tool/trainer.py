import argparse
import datetime
from pathlib import Path

from fvcore.common.config import CfgNode
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, Timer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler
from pytorch_lightning.trainer.states import RunningStage

__all__ = [
    "build_training_trainer",
    "build_testing_trainer",
    "setup_cfg",
    "log_time_elasped"
]


def build_training_trainer(args: argparse.Namespace, cfg: CfgNode) -> tuple[pl.Trainer, Timer]:
    """
    Build a PyTorch Lightning Trainer.
    """
    output_dir = Path(cfg.OUTPUT_DIR)

    logger = [
        TensorBoardLogger(output_dir, "tb_log"),
        CSVLogger(output_dir, "csv_log"),
    ]

    timer = Timer()
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=output_dir.joinpath("regular_ckpts"),
            filename="ckpt-{epoch}-{" + cfg.TRAINER.CHECKPOINT.MONITOR + ":.2f}" if cfg.TRAINER.CHECKPOINT.MONITOR is not None else "{epoch}",
            monitor="epoch",
            mode="max",
            save_top_k=5,
            every_n_epochs=cfg.TRAINER.CHECKPOINT.EVERY_N_EPOCHS,
        ),
    ]
    if cfg.TRAINER.CHECKPOINT.SAVE_BEST:
        callbacks.append(ModelCheckpoint(
            dirpath=output_dir.joinpath("best_ckpts"),
            filename="ckpt-{epoch}-{" + cfg.TRAINER.CHECKPOINT.MONITOR+ ":.2f}",
            monitor=cfg.TRAINER.CHECKPOINT.MONITOR,
            mode="min" if "loss" in cfg.TRAINER.CHECKPOINT.MONITOR else "max",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1,
        ))
    if cfg.TRAINER.PROFILER is not None:
        callbacks.append(ModelSummary(max_depth=-1))

    profiler = None
    if cfg.TRAINER.PROFILER == "advanced":
        profiler = AdvancedProfiler(
            dirpath=output_dir.joinpath("profiler"),
            filename="performance_log"
        )
    elif cfg.TRAINER.PROFILER == "pytorch":
        # To visualize the profiled operations,
        # run `nvprof --profile-from-start off -o <path_to_performance_log>` 
        profiler = PyTorchProfiler(
            dirpath=output_dir.joinpath("profiler"),
            filename="performance_log",
            emit_nvtx=True,
        )
    elif cfg.TRAINER.PROFILER is not None:
        raise ValueError(f"Unknown profiler: {cfg.TRAINER.PROFILER}")

    trainer = pl.Trainer(
        accelerator="auto",
        strategy="ddp" if args.num_gpus > 1 else "auto",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        precision="16-mixed" if cfg.TRAINER.MIXED_PRECISION else "32-true",
        logger=logger,
        callbacks=callbacks,
        max_epochs=1 if cfg.TRAINER.PROFILER else cfg.TRAINER.MAX_EPOCHS,
        check_val_every_n_epoch=cfg.TRAINER.CHECK_VAL_EVERY_N_EPOCHS,
        log_every_n_steps=cfg.TRAINER.LOG_EVERY_N_STEPS,
        accumulate_grad_batches=cfg.TRAINER.ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=cfg.TRAINER.CLIP_GRAD.VALUE if cfg.TRAINER.CLIP_GRAD.ENABLED else None,
        gradient_clip_algorithm=cfg.TRAINER.CLIP_GRAD.ALGORITHM if cfg.TRAINER.CLIP_GRAD.ENABLED else None,
        deterministic=cfg.TRAINER.DETERMINISTIC,
        benchmark=cfg.TRAINER.BENCHMARK,
        profiler=profiler,
        detect_anomaly=cfg.TRAINER.DETECT_ANOMALY,
        sync_batchnorm=cfg.TRAINER.SYNC_BATCHNORM,
        default_root_dir=output_dir,
    )

    return trainer, timer


def build_testing_trainer(args: argparse.Namespace, cfg: CfgNode) -> tuple[pl.Trainer, Timer]:
    """
    Build a PyTorch Lightning Trainer for testing.
    """
    output_dir = Path(cfg.OUTPUT_DIR)

    timer = Timer()

    trainer = pl.Trainer(
        precision="16-mixed" if cfg.TRAINER.MIXED_PRECISION else "32-true",
        logger=[TensorBoardLogger(output_dir, "tb_log")],
        callbacks=[timer],
        deterministic=cfg.TRAINER.DETERMINISTIC,
        benchmark=cfg.TRAINER.BENCHMARK,
        default_root_dir=output_dir,
    )

    return trainer, timer


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
