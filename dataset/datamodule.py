from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader

from .build import build_dataset

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
            self.train_dataset = build_dataset(self.cfg, RunningStage.TRAINING)
            self.validation_dataset = build_dataset(self.cfg, RunningStage.VALIDATING)

        if stage in ("test", "predict"):
            self.test_dataset = build_dataset(self.cfg, RunningStage.TESTING)

    def train_dataloader(self) -> DataLoader:
        cfg = self.cfg.DATALOADER
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True if self.train_dataset.sampler is None else False,
            sampler=self.train_dataset.sampler,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=cfg.DROP_LAST,
            persistent_workers=cfg.PERSISTENT_WORKERS,
        )

    def val_dataloader(self) -> DataLoader:
        cfg = self.cfg.DATALOADER
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=cfg.VAL.BATCH_SIZE if cfg.VAL.BATCH_SIZE > 0 else cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            sampler=self.validation_dataset.sampler,
            num_workers=cfg.VAL.NUM_WORKERS if cfg.VAL.NUM_WORKERS > 0 else cfg.TRAIN.NUM_WORKERS,
            collate_fn=self.validation_dataset.collate_fn,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=False,
            persistent_workers=cfg.PERSISTENT_WORKERS,
        )

    def test_dataloader(self) -> DataLoader:
        cfg = self.cfg.DATALOADER
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 0 else cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            sampler=self.test_dataset.sampler,
            num_workers=cfg.TEST.NUM_WORKERS if cfg.TEST.NUM_WORKERS > 0 else cfg.TRAIN.NUM_WORKERS,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=False,
            persistent_workers=cfg.PERSISTENT_WORKERS,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
