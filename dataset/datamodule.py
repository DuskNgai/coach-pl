from fvcore.common.config import CfgNode
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader

from .build import build_dataset

__all__ = ["BaseDataModule"]


class BaseDataModule(pl.LightningDataModule):
    """
    A general purpose data module.
    """

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()

        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit":
            self.train_dataset = build_dataset(self.cfg, RunningStage.TRAINING)
            self.validation_dataset = build_dataset(self.cfg, RunningStage.VALIDATING)

        if stage in ("test", "predict"):
            self.test_dataset = build_dataset(self.cfg, RunningStage.TESTING)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            shuffle=True,
            sampler=self.train_dataset.sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            drop_last=self.cfg.DATALOADER.DROP_LAST,
            persistent_workers=self.cfg.DATALOADER.PERSISTENT_WORKERS,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            shuffle=False,
            sampler=self.validation_dataset.sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.validation_dataset.collate_fn,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            drop_last=False,
            persistent_workers=self.cfg.DATALOADER.PERSISTENT_WORKERS,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            shuffle=False,
            sampler=self.test_dataset.sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            drop_last=False,
            persistent_workers=self.cfg.DATALOADER.PERSISTENT_WORKERS,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
