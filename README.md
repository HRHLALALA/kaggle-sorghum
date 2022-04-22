# kaggle-sorghum

## Environment
* pytorch 1.9.1/cu113
* wandb==0.12.14
* tqdm==4.64.0
* pytorch-lightning==1.5.10
* pandas==1.4.2
* timm==0.5.4
* opencv-python
* numpy
* albumentations

## Dataset Structure
* $YOUR_ROOT_DIR/sorghum-id-fgvc-9
  - train_images
    - *.png
  - test
    - *.png

## Training
Example: 
`python main.py --model_name=dm_nfnet_f0 --precision=32 --batch_size=24 --path=sorghum-id-fgvc-9/`

**Resume training**: `python main.py --model_name=dm_nfnet_f0 --precision=32 --batch_size=24 --resume_from_checkpoint={saved model path}`

## Inference
`python main.py --model_name={timm model name}  --path=$YOUR_DIR/sorghum-id-fgvc-9/ --test`
