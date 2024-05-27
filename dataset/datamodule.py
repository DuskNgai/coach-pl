import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader

from configuration import CfgNode
from dataset.build import build_dataset

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode) -> None:
        super().__init__()

        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = build_dataset(self.cfg, RunningStage.TRAINING)
            self.validation_dataset = build_dataset(self.cfg, RunningStage.VALIDATING)

        if stage in (None, "test"):
            self.test_dataset = build_dataset(self.cfg, RunningStage.TESTING)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.DATASET.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            persistent_workers=self.cfg.DATALOADER.PERSISTENT_WORKERS,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.DATASET.VALIDATION.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.validation_dataset.collate_fn,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            persistent_workers=self.cfg.DATALOADER.PERSISTENT_WORKERS,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.DATASET.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            persistent_workers=self.cfg.DATALOADER.PERSISTENT_WORKERS,
        )
