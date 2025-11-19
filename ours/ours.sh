python /data1/pinci/code/mllms_fine_grained/ours/main.py \
  --model_path /data1/pinci/ckpt/huggingface//Qwen2.5-VL-3B-Instruct \
  --annotation_path /data1/pinci/datasets/zoom_eye_data \
  --benchmark vstar \
  --answer_tag prompt_learning_partial_entropy_qwen25-vl-3b-bf16_T5 \
  --max_pixels 5500000