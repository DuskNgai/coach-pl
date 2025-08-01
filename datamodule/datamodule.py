from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader

from .dataset import build_dataset
from .sampler import build_sampler

__all__ = ["BaseDataModule"]


class BaseDataModule(pl.LightningDataModule):
    """
    A general purpose datamodule.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self._setup_stage(RunningStage.TRAINING)
            self._setup_stage(RunningStage.VALIDATING)

        if stage in ("test", "predict"):
            self._setup_stage(RunningStage.TESTING)

    def _setup_stage(self, stage: RunningStage) -> None:
        """
        Setup the dataset, sampler, and batch_sampler for the given stage.

        Args:
            stage (`RunningStage`): The current stage.

        Notes:
            The attributes set are: <stage>_dataset, <stage>_sampler, and <stage>_batch_sampler.
        """
        prefix = stage.dataloader_prefix

        setattr(self, f"{prefix}_dataset", build_dataset(self.cfg, stage))

        if self.cfg.DATAMODULE.SAMPLER.NAME is not None:
            if "BatchSampler" in self.cfg.DATAMODULE.SAMPLER.NAME:
                setattr(self, f"{prefix}_batch_sampler", build_sampler(self.cfg, stage, getattr(self, f"{prefix}_dataset")))
            elif "Sampler" in self.cfg.DATAMODULE.SAMPLER.NAME:
                setattr(self, f"{prefix}_sampler", build_sampler(self.cfg, stage, getattr(self, f"{prefix}_dataset")))

    def train_dataloader(self) -> DataLoader:
        cfg = self.cfg.DATAMODULE.DATALOADER
        dataloader_params = dict(
            dataset=self.train_dataset,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=cfg.PIN_MEMORY,
            persistent_workers=cfg.PERSISTENT_WORKERS,
        )

        if hasattr(self, "train_batch_sampler"):
            dataloader_params.update(dict(batch_sampler=self.train_batch_sampler))
        else:
            # No batch sampler, then `batch_size` and `drop_last` should be set.
            dataloader_params.update(dict(
                batch_size=cfg.TRAIN.BATCH_SIZE,
                drop_last=cfg.DROP_LAST,
            ))

            # When `sampler` is set, `shuffle` should be set to `False`.
            if hasattr(self, "train_sampler"):
                dataloader_params.update(dict(shuffle=False, sampler=self.train_sampler))
            else:
                dataloader_params.update(dict(shuffle=cfg.TRAIN.SHUFFLE))

        return DataLoader(**dataloader_params)

    def val_dataloader(self) -> DataLoader:
        cfg = self.cfg.DATAMODULE.DATALOADER
        dataloader_params = dict(
            dataset=self.val_dataset,
            num_workers=cfg.VAL.NUM_WORKERS if cfg.VAL.NUM_WORKERS > 0 else cfg.TRAIN.NUM_WORKERS,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=cfg.PIN_MEMORY,
            persistent_workers=cfg.PERSISTENT_WORKERS,
        )

        if hasattr(self, "val_batch_sampler"):
            dataloader_params.update(dict(batch_sampler=self.val_batch_sampler))
        else:
            # No batch sampler, then `batch_size` and `drop_last` should be set,
            # and `shuffle` is always `False` in val.
            dataloader_params.update(
                dict(
                    batch_size=cfg.VAL.BATCH_SIZE if cfg.VAL.BATCH_SIZE > 0 else cfg.TRAIN.BATCH_SIZE,
                    shuffle=False,
                    sampler=getattr(self, "val_sampler", None),
                    drop_last=False,
                )
            )

        return DataLoader(**dataloader_params)

    def test_dataloader(self) -> DataLoader:
        cfg = self.cfg.DATAMODULE.DATALOADER
        dataloader_params = dict(
            dataset=self.test_dataset,
            num_workers=cfg.TEST.NUM_WORKERS if cfg.TEST.NUM_WORKERS > 0 else cfg.TRAIN.NUM_WORKERS,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=cfg.PIN_MEMORY,
            persistent_workers=cfg.PERSISTENT_WORKERS,
        )

        # Similar to `val_dataloader`.
        if hasattr(self, "test_batch_sampler"):
            dataloader_params.update(dict(batch_sampler=self.test_batch_sampler))
        else:
            dataloader_params.update(
                dict(
                    batch_size=cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 0 else cfg.TRAIN.BATCH_SIZE,
                    shuffle=False,
                    sampler=getattr(self, "test_sampler", None),
                    drop_last=False,
                )
            )

        return DataLoader(**dataloader_params)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
