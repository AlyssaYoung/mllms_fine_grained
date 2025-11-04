CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/pinci/code/mllms_fine_grained/test_qwen25_vl.py \
  --model_path /root/autodl-tmp/ckpt//Qwen2.5-VL-3B-Instruct \
  --annotation_path /root/autodl-tmp/dataset/zoom_eye_data/zoom_eye_data \
  --benchmark vstar \
  --answer_tag test_qwen25-vl-3b-bf16\
  --multi_gpu