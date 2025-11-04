CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /home/pinci/code/mllms_fine_grained/prompt_learning.py \
  --model_path /data1/pinci/ckpt/huggingface//Qwen2.5-VL-3B-Instruct \
  --annotation_path /data1/pinci/datasets/zoom_eye_data \
  --benchmark vstar \
  --answer_tag prompt_learning_qwen25-vl-3b-bf16 \
  --multi_gpu