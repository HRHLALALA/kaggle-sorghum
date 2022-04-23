import os

import albumentations as A
import pandas as pd
import torch
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from config import CFG
from data import get_loader
from models import create_model
import logging
import argparse

os.environ['TORCH_HOME'] = "./pretrain"
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logger = logging.getLogger("pytorch_lightning.core")


if os.environ['TRAIN_LOGGER']  is None or os.environ['TRAIN_LOGGER'] == "wandb":
    from pytorch_lightning.loggers import WandbLogger
    logger.info("Use WANDB logger")
    os.environ['TRAIN_LOGGER'] = "wandb"

elif os.environ['TRAIN_LOGGER'] == "tb":
    from pytorch_lightning.loggers import TensorBoardLogger
    logger.info("Use tensorboard logger")
else:
    raise NotImplementedError("Not implemented " + os.environ['TRAIN_LOGGER'])



def get_transform(phase: str, img_size):
    if phase == 'train':
        return Compose([
            A.RandomResizedCrop(height=img_size, width= img_size),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Blur(p=0.1),
                A.GaussianBlur(p=0.1),
                A.MotionBlur(p=0.1),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(p=0.1),
                A.ISONoise(p=0.1),
                A.GridDropout(ratio=0.5, p=0.2),
                A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
def main(cfg):
    
    """
    Prepare Data
    """
    PATH = cfg.path

    TRAIN_DIR = os.path.join(PATH,'train_images/')
    TEST_DIR = os.path.join(PATH, 'test/')
    

    df_all = pd.read_csv(os.path.join(PATH, "train_cultivar_mapping.csv"))
    logger.info(len(df_all))
    df_all.dropna(inplace=True)
    logger.info(len(df_all))
    df_all.head()
    unique_cultivars = list(df_all["cultivar"].unique())
    num_classes = len(unique_cultivars)

    cfg.num_classes = num_classes
    logger.info(num_classes)

    df_all["file_path"] = df_all["image"].apply(lambda image: os.path.join(TRAIN_DIR,image))
    df_all["cultivar_index"] = df_all["cultivar"].map(lambda item: unique_cultivars.index(item))
    df_all["is_exist"] = df_all["file_path"].apply(lambda file_path: os.path.exists(file_path))
    df_all = df_all[df_all.is_exist==True]
    df_all.head()
    model = create_model(cfg)
    trainer = None
    pl_logger = None
    if not cfg.test:
        skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

        for train_idx, valid_idx in skf.split(df_all['image'], df_all["cultivar_index"]):
            df_train = df_all.iloc[train_idx]
            df_valid = df_all.iloc[valid_idx]

        logger.info(f"train size: {len(df_train)}")
        logger.info(f"valid size: {len(df_valid)}")

        logger.info(df_train.cultivar.value_counts())
        logger.info(df_valid.cultivar.value_counts())

        train_dset, train_loader = get_loader(df_train, get_transform('train', cfg.img_size), batch_size=cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=12)
        val_dset, valid_loader = get_loader(df_valid, get_transform('valid',cfg.img_size), batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=12)

        cfg.steps_per_epoch = len(train_loader)
        logger.info(cfg.steps_per_epoch)


        if os.environ['TRAIN_LOGGER'] == "wandb":
            pl_logger = WandbLogger(name="kaggle-sorghum", save_dir='logs/' + cfg.model_name, log_model=True)
        else:
            pl_logger = TensorBoardLogger(name="kaggle-sorghum", save_dir='logs/' + cfg.model_name)

        os.makedirs("logs/"+cfg.model_name + "/wandb/", exist_ok=True)

        pl_logger.log_hyperparams(cfg)
        checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                              save_top_k=1,
                                              save_last=True,
                                              save_weights_only=True,
                                              filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
                                              verbose=False,
                                              mode='min')

        trainer = Trainer(
            max_epochs=cfg.num_epochs,
            gpus=-1,
            accumulate_grad_batches=cfg.accum,
            precision=cfg.precision,
            accelerator="ddp",
            callbacks=[checkpoint_callback],
            logger=pl_logger,
            weights_summary='top',
            sync_batchnorm=True,
       )

        trainer.fit(model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader,
                    ckpt_path=cfg.resume_from_checkpoint)


    sub = pd.read_csv(os.path.join(PATH,"sample_submission.csv"))
    sub.head()

    sub["file_path"] = sub["filename"].apply(lambda image: TEST_DIR + image)
    sub["cultivar_index"] = 0
    sub.head()

    test_dset,test_loader = get_loader(sub, get_transform('valid',cfg.img_size), batch_size=cfg.batch_size, shuffle=False, num_workers=12)

    model.cuda()
    model.eval()
    if trainer is None:
        trainer = Trainer(gpus=[0], logger=False)
        predictions = trainer.predict(model,test_loader)
    else:
        predictions = trainer.predict(dataloaders = test_loader, ckpt_path="best")

    tmp = predictions[0]
    for i in range(len(predictions) - 1):
        tmp = torch.cat((tmp, predictions[i + 1]))

    predictions = [unique_cultivars[pred] for pred in tmp]
    sub = pd.read_csv(os.path.join(PATH,"sample_submission.csv"))
    sub["cultivar"] = predictions
    if pl_logger is not None and os.environ['TRAIN_LOGGER'] =="wandb":
        pl_logger.log_table(key="submission", dataframe=sub)
    else:
        logger.warning("Warning: no log_table methods. Save to submission.csv only")
    sub.to_csv('submission.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="sorghum-id-fgvc-9/")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--test", action="store_true")
    parser = CFG.add_parser(parser)
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    """
    Adjust some augments
    """
    args.lr *= (torch.cuda.device_count() *  args.batch_size /16)
    # CFG.model_name = args.model_name
    # CFG.path = args.path
    # CFG.resume_from_checkpoint = args.resume_from_checkpoint
    # CFG.test = args.test
    main(args)
