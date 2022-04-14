# kaggle-sorghum

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
