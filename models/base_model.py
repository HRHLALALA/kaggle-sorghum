import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import torchmetrics

from utils import test_time_augmentation


class BaseModel(pl.LightningModule):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=cfg.num_classes)
        self.metric = torchmetrics.Accuracy(threshold=0.5, num_classes=self.cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.cfg.lr

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             epochs=self.cfg.num_epochs,
                                                             steps_per_epoch=self.cfg.steps_per_epoch,
                                                             max_lr=self.cfg.max_lr, pct_start=self.cfg.pct_start,
                                                             div_factor=self.cfg.div_factor,
                                                             final_div_factor=self.cfg.final_div_factor)

        scheduler = {'scheduler': self.scheduler, 'interval': 'step', }

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target'].long()
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.metric(output.argmax(1), target)
        logs = {'train_loss': loss, 'train_acc': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target'].long()
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.metric(output.argmax(1), target)
        logs = {'valid_loss': loss, 'valid_acc': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target'].long()
        output = test_time_augmentation(self.model, image)
        loss = self.criterion(output, target)
        score = self.metric(output.argmax(1), target)
        logs = {'valid_loss': loss, 'valid_acc': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # outputs = self.model(batch['image'])
        outputs = test_time_augmentation(self.model, batch['image'])
        preds = outputs.detach().cpu()
        return preds.argmax(1)
