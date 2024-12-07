import argparse
import datetime
from pathlib import Path

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, Timer, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

from coach_pl.configuration import CfgNode
from coach_pl.utils.collect_env import collect_env_info

__all__ = [
    "build_training_trainer",
    "build_testing_trainer",
    "setup_cfg",
    "log_time_elasped",
]


def build_training_trainer(args: argparse.Namespace, cfg: DictConfig) -> tuple[pl.Trainer, Timer]:
    """
    Build a PyTorch Lightning Trainer.
    """
    output_dir = Path(cfg.OUTPUT_DIR)

    strategy = cfg.TRAINER.STRATEGY
    if args.num_gpus * args.num_nodes == 1 and strategy != "auto":
        rank_zero_warn("Single GPU training is not supported for the specified strategy. Automatically set to 'auto'.")
        strategy = "auto"

    if "deepspeed" in strategy:
        StrategyRegistry[strategy]["init_params"].update({
            "logging_batch_size_per_gpu": cfg.DATALOADER.BATCH_SIZE
        })

    logger = [
        CSVLogger(save_dir=output_dir, name="csv_log"),
        TensorBoardLogger(save_dir=output_dir, name="tb_log"),
    ]

    timer = Timer()
    callbacks = [
        timer,
        TQDMProgressBar(refresh_rate=cfg.TRAINER.LOG_EVERY_N_STEPS),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=2),
    ]
    if cfg.TRAINER.CHECKPOINT.MONITOR is not None:
        assert str(cfg.TRAINER.CHECKPOINT.MONITOR).find('/') == -1, "Monitor should not contain `/`"
        filename = "{epoch}-{" + cfg.TRAINER.CHECKPOINT.MONITOR + ":.4g}"
    else:
        filename = "{epoch}"

    if cfg.TRAINER.CHECKPOINT.SAVE_BEST:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir.joinpath("best_ckpts"),
                filename=filename,
                monitor=cfg.TRAINER.CHECKPOINT.MONITOR,
                save_last="link",
                save_top_k=3,
                mode=cfg.TRAINER.CHECKPOINT.MONITOR_MODE,
                every_n_epochs=1,
            )
        )
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir.joinpath("regular_ckpts"),
            filename=filename,
            monitor="epoch",
            save_last="link",
            save_top_k=5,
            mode="max",
            every_n_epochs=cfg.TRAINER.CHECKPOINT.EVERY_N_EPOCHS,
        )
    )

    if cfg.TRAINER.PROFILER is None:
        profiler = None
    elif cfg.TRAINER.PROFILER == "simple":
        profiler = SimpleProfiler(dirpath=output_dir.joinpath("profiler"), filename="performance_log")
    elif cfg.TRAINER.PROFILER == "advanced":
        profiler = AdvancedProfiler(dirpath=output_dir.joinpath("profiler"), filename="performance_log")
    elif cfg.TRAINER.PROFILER == "pytorch":
        # To visualize the profiled operations,
        # run `nvprof --profile-from-start off -o <path_to_performance_log>`
        profiler = PyTorchProfiler(
            dirpath=output_dir.joinpath("profiler"),
            filename="performance_log",
            emit_nvtx=True,
        )
    else:
        raise KeyError(f"Unknown profiler: {cfg.TRAINER.PROFILER}")

    trainer = pl.Trainer(
        accelerator="auto",
        strategy=strategy,
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        precision=cfg.TRAINER.PRECISION,
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.TRAINER.MAX_EPOCHS if cfg.TRAINER.PROFILER is None else 1,
        check_val_every_n_epoch=cfg.TRAINER.CHECKPOINT.EVERY_N_EPOCHS,
        num_sanity_val_steps=cfg.TRAINER.NUM_SANITY_VAL_STEPS if cfg.TRAINER.PROFILER is None else 0,
        log_every_n_steps=cfg.TRAINER.LOG_EVERY_N_STEPS,
        accumulate_grad_batches=cfg.TRAINER.ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=cfg.TRAINER.CLIP_GRAD.VALUE,
        gradient_clip_algorithm=cfg.TRAINER.CLIP_GRAD.ALGORITHM,
        deterministic=cfg.TRAINER.DETERMINISTIC,
        benchmark=cfg.TRAINER.BENCHMARK,
        profiler=profiler,
        detect_anomaly=cfg.TRAINER.DETECT_ANOMALY,
        sync_batchnorm=cfg.TRAINER.SYNC_BATCHNORM,
        default_root_dir=output_dir,
    )

    return trainer, timer


def build_testing_trainer(cfg: DictConfig) -> tuple[pl.Trainer, Timer]:
    """
    Build a PyTorch Lightning Trainer for testing.
    """
    output_dir = Path(cfg.OUTPUT_DIR)

    timer = Timer()

    trainer = pl.Trainer(
        precision=cfg.TRAINER.PRECISION,
        logger=[TensorBoardLogger(output_dir, "tb_log")],
        callbacks=[timer],
        deterministic=cfg.TRAINER.DETERMINISTIC,
        benchmark=cfg.TRAINER.BENCHMARK,
        default_root_dir=output_dir,
    )

    return trainer, timer


def setup_cfg(args: argparse.Namespace) -> DictConfig:
    """
    Create configs from default settings, file, and command-line arguments.
    """
    cfg = CfgNode.load_yaml_with_base(args.config_file)
    CfgNode.merge_with_dotlist(cfg, args.opts)
    CfgNode.set_readonly(cfg, True)
    return cfg


@rank_zero_only
def log_configurations(cfg: DictConfig) -> None:
    print(collect_env_info())
    print(CfgNode.to_yaml(cfg))


@rank_zero_only
def log_time_elasped(timer: Timer, stage: RunningStage) -> None:
    elasped_time = datetime.timedelta(seconds=timer.time_elapsed(stage))
    print(f"Running time for {stage}: {elasped_time}")
