set -e
python main.py --resume_from_checkpoint=logs/tf_efficientnetv2_xl_in21k/kaggle-sorghum/version_1/checkpoints/epoch=36-valid_loss=0.0204-valid_acc=0.9933.ckpt --submit --model_name=tf_efficientnetv2_xl_in21k --test_time_augmentation --batch_size=16
kaggle competitions submit -c sorghum-id-fgvc-9 -f submission.csv -m "Message"
