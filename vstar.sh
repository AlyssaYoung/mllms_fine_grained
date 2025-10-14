python /root/autodl-tmp/code/mllms_fine_grained/main.py \
  --model_path /root/autodl-tmp/ckpt/llava-v1.5-7b \
  --vision_tower_path /root/autodl-tmp/ckpt/clip-vit-large-patch14-336 \
  --annotation_path /root/autodl-tmp/dataset/zoom_eye_data/zoom_eye_data \
  --benchmark vstar \
  --answer_tag vstar_divprune_pos_emb_float_0.01 \
  --resized_res 672 \
  --multi_gpu