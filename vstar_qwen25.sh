python /home/pinci/code/mllms_fine_grained/test_qwen25_vl.py \
  --model_path /data1/pinci/ckpt/huggingface/Qwen2.5-VL-3B-Instruct\
  --annotation_path /data1/pinci/datasets/zoom_eye_data/ \
  --benchmark vstar \
  --answer_tag test_qwen25-vl-3b_f32_96 \
  --multi_gpu