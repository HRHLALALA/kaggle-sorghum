set -e
python main.py --resume_from_checkpoint=logs/seresnext26tn_32x4d/3awm4320/checkpoints/epoch=37-valid_loss=0.0190-valid_acc=0.9939.ckpt --submit --model_name=seresnext26tn_32x4d --batch_size=64
kaggle competitions submit -c sorghum-id-fgvc-9 -f submission.csv -m "seresnext26tn_32x4d fold 3"
