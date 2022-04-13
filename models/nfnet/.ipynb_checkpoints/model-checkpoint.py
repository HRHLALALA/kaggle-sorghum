import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import timm
from .agc import AGC

class NFNet(pl.LightningModule):
    def __init__(self, model_name, cfg,pretrained=True):
        super(NFNet, self).__init__()
        assert "nfnet" in model_name, "Must use NFNet checkpoints"
        self.cfg = cfg
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=self.cfg.num_classes, drop_rate=0.5)
        self.metric = torchmetrics.Accuracy(threshold=0.5, num_classes=self.cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.cfg.lr

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = AGC(
            self.model.parameters(), 
            torch.optim.Adam(self.model.parameters(), lr=self.lr),
            model = self.model,
            ignore_agc=['head']
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             epochs=self.cfg.num_epochs, steps_per_epoch=self.cfg.steps_per_epoch,
                                                             max_lr=self.cfg.max_lr, pct_start=self.cfg.pct_start, 
                                                             div_factor=self.cfg.div_factor, final_div_factor=self.cfg.final_div_factor)
        scheduler = {'scheduler': self.scheduler, 'interval': 'step',}

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
            on_step=False, on_epoch=True, prog_bar=True, logger=True
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
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
