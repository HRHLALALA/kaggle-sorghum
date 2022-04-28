import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from losses.arc_face_loss import ArcMarginProduct_subcenter, Swish_module
from models.base_model import BaseModel


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class HybridSwinModel(nn.Module):
    def __init__(self, backbone, embedder, img_size, num_classes,embedding_size=512, pretrained=True, arc_face_head=False, neck_option="option-D"):
        super(HybridSwinModel, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embedder = timm.create_model(embedder, features_only=True, out_indices=[2], pretrained=pretrained)
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=img_size, embed_dim=128)
        self.n_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        self.embedding_size = embedding_size
        if neck_option == "option-N":
            self.neck = nn.Sequential(
                nn.Linear(self.n_features, self.embedding_size, bias=True),
                torch.nn.PReLU()
            )
        if neck_option == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.n_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif neck_option == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.n_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif neck_option == "option-X":
            self.neck = nn.Sequential(
                nn.Linear(self.n_features, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )

        elif neck_option == "option-S":
            self.neck = nn.Sequential(
                nn.Linear(self.n_features, self.embedding_size),
                Swish_module()
            )
            
        if arc_face_head:
            self.fc = ArcMarginProduct_subcenter( self.embedding_size, num_classes)
        else:
            self.fc = nn.Linear( self.embedding_size, num_classes)

    def forward(self, images):
        features = self.backbone(images)              # features = (bs, embedding_size)
        features = self.neck(features)
        output = self.fc(features)                    # outputs  = (bs, num_classes)
        return output

class HybridSwin(BaseModel):
    def __init__(self, cfg):
        super(HybridSwin, self).__init__(cfg, False)
        assert "efficientnet" in cfg.model_name, "Must be EfficientNet related checkpoints to build the embedder"
        assert "swin" in cfg.model_name, "Must be Swin related checkpoints to build the embedder"
        backbone_name, embedder_name = cfg.model_name.split('.')
        self.model = HybridSwinModel(
            backbone=backbone_name,
            embedder=embedder_name,
            img_size=cfg.img_size,
            num_classes=cfg.num_classes,
            arc_face_head = cfg.arc_face_head,
            neck_option = cfg.neck_option
        )

