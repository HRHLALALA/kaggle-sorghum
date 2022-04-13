import torch
class CFG:
    seed = 42
    model_name = 'dm_nfnet_f0'
    pretrained = True
    img_size = 512
    num_classes = 100
    lr = 2e-4
    max_lr = 1e-3
    pct_start = 0.2
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    num_epochs = 40
    batch_size = 36
    accum = 1
    precision = 16
    n_fold = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')