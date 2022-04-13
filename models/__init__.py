def create_model(model_name, cfg):
    if "nfnet" in model_name:
        from models.nfnet import NFNet
        Model = NFNet
    elif "efficientnet" in model_name:
        from models.efficientnet import EfficientNet
        Model = EfficientNet
    else:
        raise NotImplementedError(f"No model {model_name}")

    if cfg.resume_from_checkpoint is not None:
        return Model.load_from_checkpoint(checkpoint_path = cfg.resume_from_checkpoint)
    else:
        return Model(model_name, cfg, cfg.pretrained)
