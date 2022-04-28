set -e
python main.py --resume_from_checkpoint=logs/swin_base_patch4_window7_224_in22k.tf_efficientnet_b4_ns/11dhn6da/checkpoints/epoch=37-valid_loss=0.0930-valid_acc=0.9645.ckpt --submit --model_name=swin_base_patch4_window7_224_in22k.tf_efficientnet_b4_ns --batch_size=32 --num_workers=16  --img_size=448
kaggle competitions submit -c sorghum-id-fgvc-9 -f submission.csv -m "swin_base_patch4_window7_224_in22k.tf_efficientnet_b4_ns fold 2 resized"
