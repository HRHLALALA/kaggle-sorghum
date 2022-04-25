import torch

def test_time_augmentation(model, x, aug_kinds=10):
    enhance_factor = torch.rand(x.shape[0], aug_kinds, 1, 1, 1, device=x.device) * 2
    x = x[:,None]
    enh_x = torch.cat([enhance_factor * x,x],dim=1).reshape(-1, *x.shape[-3:])
    preds = model(enh_x)
    preds = preds.reshape(-1, aug_kinds + 1,preds.shape[-1]).mean(dim=1)
    return preds



