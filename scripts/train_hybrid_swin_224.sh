python main.py \
    --model_name=swin_base_patch4_window7_224_in22k.tf_efficientnet_b4_ns \
    --batch_size=32 \
    --fold_id=0 \
    --num_workers=12 \
    --img_size=448 \
    --loss="arcface" \
    --arc_face_head \
    --neck_option option-D
