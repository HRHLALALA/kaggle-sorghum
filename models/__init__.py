import warnings
def create_model(cfg):
    if "nfnet" in cfg.model_name:
        from models.nfnet import NFNet
        Model = NFNet
    elif "efficientnet" in cfg.model_name:
        from models.efficientnet import EfficientNet
        Model = EfficientNet
    else:
        from models.base_model import BaseModel
        Model = BaseModel
        warnings.warn(f"Not implemented model {cfg.model_name}")

    if cfg.resume_from_checkpoint is not None:
        return Model.load_from_checkpoint(cfg.resume_from_checkpoint, cfg=cfg)
    else:
        return Model(cfg)
