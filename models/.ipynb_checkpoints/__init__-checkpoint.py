def create_model(model_name, cfg):
    if "nfnet" in model_name:
        from models.nfnet import NFNet
        return NFNet(model_name, cfg)
    elif "efficientnet" in model_name:
        from models.efficientnet import EfficientNet
        return EfficientNet(model_name, cfg)