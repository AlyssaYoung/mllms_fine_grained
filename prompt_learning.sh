# python /home/pinci/code/mllms_fine_grained/prompt_learning.py \
#   --model_path /data1/pinci/ckpt/huggingface//Qwen2.5-VL-7B-Instruct \
#   --annotation_path /data1/pinci/datasets/zoom_eye_data \
#   --benchmark vstar \
#   --answer_tag prompt_learning_qwen25-vl-7b-bf16 \
#   --max_pixels 4000000 \
#   --multi_gpu

python /home/pinci/code/mllms_fine_grained/prompt_learning.py \
  --model_path /data1/pinci/ckpt/huggingface//Qwen2.5-VL-3B-Instruct \
  --annotation_path /data1/pinci/datasets/zoom_eye_data \
  --benchmark vstar \
  --answer_tag prompt_learning_output_attn_qwen25-vl-3b-bf16 \
  --max_pixels 5500000 \
  --multi_gpu \
  --attn_mode output