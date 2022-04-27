import torch

import kornia.augmentation as A


def normal_test(model, x, cfg):
    aug_list = A.AugmentationSequential(
        A.Resize((cfg.img_size, cfg.img_size), p=1),
        data_keys=["input"],
        same_on_batch=False,
    )

    preds = model(aug_list(x))
    return preds

def test_time_augmentation(model, x, cfg, aug_kinds=10):
    aug_list = A.AugmentationSequential(
        A.RandomResizedCrop((cfg.img_size, cfg.img_size), p =1),
        A.RandomHorizontalFlip(),
        A.RandomVerticalFlip(),
        # A.RandomAffine(20, [0.1, 0.1], [0.7, 1.2], [30., 50.], p=1.0),
        data_keys=["input"],
        same_on_batch=False,
    )
    enh_x = aug_list(x[:,None].expand(-1,aug_kinds, -1,-1,-1).reshape(-1, *x.shape[-3:]))\
        .reshape(-1,aug_kinds,3, cfg.img_size, cfg.img_size)

    # enhance_factor = torch.rand(x.shape[0], aug_kinds, 1, 1, 1, device=x.device) * 2
    # x = x[:,None]
    x = A.Resize((cfg.img_size, cfg.img_size), p=1)(x)
    enh_x = torch.cat([enh_x,x[:,None]],dim=1).reshape(-1, *x.shape[-3:])
    preds = model(enh_x)
    preds = preds.reshape(-1, aug_kinds + 1,preds.shape[-1])

    return torch.softmax(preds,dim=-1).mean(dim=1)



