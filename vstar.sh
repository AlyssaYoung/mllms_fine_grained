python /data1/pinci/code/mllms_fine_grained/main.py \
  --model_path /data1/pinci/ckpt/huggingface/llava-v1.5-7b \
  --vision_tower_path /data1/pinci/ckpt/huggingface/clip-vit-large-patch14-336 \
  --annotation_path /data1/pinci/datasets/zoom_eye_data \
  --benchmark vstar \
  --answer_tag test_multigpu \
  --resized_res 672 \
  --multi_gpu