# python /data1/pinci/code/mllms_fine_grained/main_divprune.py \
#   --model_path /data1/pinci/ckpt/huggingface/llava-v1.5-7b \
#   --vision_tower_path /data1/pinci/ckpt/huggingface/clip-vit-large-patch14-336 \
#   --annotation_path /data1/pinci/datasets/zoom_eye_data \
#   --benchmark hr-bench_4k \
#   --answer_tag hr_4k_divprune_combine \
#   --resized_res 672 \
#   --multi_gpu

python /data1/pinci/code/mllms_fine_grained/main_divprune.py \
  --model_path /data1/pinci/ckpt/huggingface/llava-v1.5-7b \
  --vision_tower_path /data1/pinci/ckpt/huggingface/clip-vit-large-patch14-336 \
  --annotation_path /data1/pinci/datasets/zoom_eye_data \
  --benchmark hr-bench_4k \
  --answer_tag hr_4k_baseline \
  --resized_res 336 \
  --multi_gpu