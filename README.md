# kaggle-sorghum

## Dataset Structure
* $YOUR_DIR/sorghum-id-fgvc-9
  - train_images
    - *.png
  - test
    - *.png

## Training
`python main.py --model_name={timm model name}  --path=$YOUR_DIR/sorghum-id-fgvc-9/ --{Any parameters in config.py}`

**Resume training**: `python main.py --model_name={timm model name} --resume_from_checkpoint={saved model path} --...`

## Inference
`python main.py --model_name={timm model name}  --path=$YOUR_DIR/sorghum-id-fgvc-9/ --test`
