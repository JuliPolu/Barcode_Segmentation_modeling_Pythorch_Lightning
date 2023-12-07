import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import Config
from losses import get_losses
from metrics import get_metrics
from utils import load_object


class SegmentModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._model = smp.create_model(**self._config.model_kwargs)
        self._losses = get_losses(self._config.losses)
        
        metrics = get_metrics(
            task = 'binary',
            average=self._config.metrics_average,
            threshold=self._config.threshold,
        )
        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """
        Считаем лосс.
        """
        images, target = batch
        images, target = images.float(), target.float()
        pr_logits = self(images)
        return self._calculate_loss(pr_logits, target, 'train_')

    def validation_step(self, batch, batch_idx):
        """
        Считаем лосс и метрики.
        """
        images, target = batch
        images, target = images.float(), target.float()
        pr_logits = self(images)
        self._calculate_loss(pr_logits, target, 'val_')
        pr_labels = torch.sigmoid(pr_logits)
        self._valid_metrics(pr_labels, target)

    def test_step(self, batch, batch_idx):
        """
        Считаем метрики.
        """
        images, target = batch
        images, target = images.float(), target.float()
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, target)

    
    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        pr_logits: torch.Tensor,
        target: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            loss = cur_loss.loss(pr_logits, target)
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss


