python /root/code/mllms_fine_grained/main.py \
  --model_path /root/autodl-tmp/ckpt/llava-v1.5-7b \
  --vision_tower_path /root/autodl-tmp/ckpt/clip-vit-large-patch14-336 \
  --annotation_path /root/dataset/zoom_eye_data/zoom_eye_data \
  --benchmark vstar \
  --answer_tag test_multigpu \
  --resized_res 672 \
  --multi_gpu