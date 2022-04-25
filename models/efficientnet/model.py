import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import torchmetrics

from models.base_model import BaseModel


class EfficientNet(BaseModel):
    def __init__(self, cfg):
        super(EfficientNet, self).__init__(cfg)
        assert "efficientnet" in cfg.model_name, "Must be EfficientNet related checkpoints"
