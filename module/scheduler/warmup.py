from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["LinearWarmupLR"]


class LinearWarmupLR(LRScheduler):
    """
    Warmup learning rate scheduler.
    Starting from a warmup factor * base learning rate, the learning rate
    is linearly increased to the base learning rate over a number of epochs.
    """

    def __init__(self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        warmup_epochs: int,
        warmup_factor: float,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.scheduler = scheduler

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch <= self.warmup_epochs:
            scale = self.warmup_factor + (1 - self.warmup_factor) * (self.last_epoch - 1) / (self.warmup_epochs - 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            return self.scheduler.get_last_lr()

    def step(self, epoch: int | None = None) -> None:
        # The scheduler has to be stepped first to ensure that the learning rate is updated.
        # The first step has already been done in the super class.
        if self.last_epoch >= 0:
            self.scheduler.step(epoch)

        super().step(epoch)
