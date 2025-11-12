from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from llava.conversation import conv_templates
from llava.constants import DEFAULT_IMAGE_TOKEN
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import pdb
import argparse
import os
import json
from tqdm import tqdm
from PIL import Image
import copy

def get_direct_response(model, processor, annotation, messages_template, image_folder):
    input_image = annotation["input_image"]
    image_path = os.path.join(image_folder, input_image)
    question = annotation["question"]
    question_input_messages = messages_template("user", image_path, question)

    text = processor.apply_chat_template(
        question_input_messages, tokenize=False, add_generation_prompt=True
    )
    print(text)
    image_inputs, video_inputs = process_vision_info(question_input_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    return output_text

@torch.inference_mode()
def multiple_choices_inference(model, processor, tokenizer, annotation, messages_template, image_folder):
    input_image = annotation["input_image"]
    image_path = os.path.join(image_folder, input_image)
    question = annotation["question"]
    options = annotation.get('options', None)

    shared_messages = messages_template(image_path, question)
    question_prompt = processor.apply_chat_template(
        shared_messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(shared_messages)

    question_inputs = processor(
        text=[question_prompt],
        images=image_inputs,
        return_tensors="pt",
    )
    question_input_ids = question_inputs.input_ids
    question_len = question_input_ids.shape[1]

    with torch.no_grad():
        output = model(
            input_ids=question_inputs.input_ids,
            attention_mask=question_inputs.attention_mask,
            pixel_values=question_inputs.pixel_values,
            image_grid_thw=question_inputs.image_grid_thw,
            return_dict=True,
            use_cache=True,
        )
        question_logits = output.logits
        past_key_values = output.past_key_values
        # 保存 rope_deltas，在模型内部会自动使用
        rope_device = next(model.parameters()).device
        model.rope_deltas = output.rope_deltas.to(rope_device)
        initial_past_key_values = copy.deepcopy(past_key_values)

    loss_list = []
    for option in options:
        past_key_values = copy.deepcopy(initial_past_key_values)  # >>> 增量相关代码

        # 在 shared_messages 基础上追加 assistant 回答
        option_messages = shared_messages + [{"role": "assistant", "content": option}]
        option_prompt = processor.apply_chat_template(
            option_messages, tokenize=False, add_generation_prompt=False
        )

        option_inputs = processor(
            text=[option_prompt],
            images=image_inputs,
            return_tensors="pt",
        )
        option_token_ids = option_inputs.input_ids[:, question_len:]
        labels = option_token_ids.view(-1)

        # 构建完整 attention_mask = 问题部分 + 选项部分
        option_attn_mask = torch.ones_like(option_token_ids, dtype=question_inputs.attention_mask.dtype)
        full_attn_mask = torch.cat([question_inputs.attention_mask, option_attn_mask], dim=1)

        # 计算增量位置，用于更新 3D RoPE
        past_len = question_len
        cache_position = torch.arange(
            past_len, past_len + option_token_ids.shape[1],
            device=option_token_ids.device,
            dtype=torch.long
        )

        # 增量前向：只输入新 token，沿用 past_key_values，传入 cache_position
        output_option = model(
            input_ids=option_token_ids,
            attention_mask=full_attn_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=False,
            return_dict=True,
            pixel_values=None,                     # 避免重复处理图像
            image_grid_thw=question_inputs.image_grid_thw,  # 保留视觉网格信息
        )

        # 对齐 logits：首个 token 用 question_logits 的最后一个预测，其余用 output_option.logits
        logits = torch.cat(
            [question_logits[:, -1:], output_option.logits[:, :-1]],
            dim=1
        ).view(-1, model.config.vocab_size)

        # 计算交叉熵
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.float(), labels)
        loss_list.append(loss)

        del past_key_values
    del initial_past_key_values

    option_chosen = torch.stack(loss_list).argmin().item()
    loss_list_out = [l.detach().float().item() for l in loss_list]
    torch.cuda.empty_cache()
    return option_chosen, loss_list_out


def load_qwen_model(model_path, multi_gpu):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, # torch_dtype=torch.bfloat16,  # Use float16 instead of auto
                device_map="auto",
                attn_implementation="flash_attention_2"  # Enable flash attention for memory efficiency
            )
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    max_pixels = 12845056
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    # processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, processor, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run qwen2.5_vl inference with ZoomEye-compatible outputs")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/ckpt/Qwen2.5-VL-7B-Instruct", help="Path to LLaVA model checkpoint")
    parser.add_argument("--answers_file", type=str, default=None, help="Path to output answers .jsonl file")
    parser.add_argument("--annotation_path", type=str, default="/root/dataset/zoom_eye_data/zoom_eye_data", help="Path to dataset root (contains benchmark folders)")
    parser.add_argument("--benchmark", type=str, choices=["vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld"], default="vstar")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--process_all_chunks", action="store_true", help="Process all chunks in a single run (ignores chunk_idx)")
    parser.add_argument("--direct_answer", action="store_true")
    parser.add_argument("--conv_type", type=str, default="qwen_2_5", help="Conversation template type")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (ignored if --multi_gpu)")
    parser.add_argument("--vision_tower_path", type=str, default=None, help="Path to vision tower weights")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU inference (device_map=auto)")
    parser.add_argument("--answer_tag", type=str, required=True, help="Answer tag")
    # parser.add_argument("--dinov3_feat_path", type=str, required=True, help="Path to dinov3 features")

    args = parser.parse_args()

    model, processor, tokenizer = load_qwen_model(args.model_path, multi_gpu=args.multi_gpu)
    # question_messages = lambda img, question: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    # option_messages = lambda img, question, option: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}, {"role": "assistant", "content": option}]
    messages = lambda img, question: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]

    if args.answers_file is None:
        answers_dir = f"eval/answers/{args.benchmark}"
        answers_dir = os.path.join(answers_dir, os.path.basename(args.model_path))
        os.makedirs(answers_dir, exist_ok=True)
        args.answers_file = os.path.join(answers_dir, f"{args.answer_tag}.jsonl")
        print(args.answers_file)

    # Load annotations and chunk
    annotation_file = os.path.join(args.annotation_path, f"{args.benchmark}/annotation_{args.benchmark}.json")
    with open(annotation_file, 'r') as f:
        all_annotations = json.load(f)

    image_folder = os.path.join(args.annotation_path, f"{args.benchmark}")

    # Determine file mode and annotations to process
    if args.process_all_chunks and args.num_chunks > 1:
        # Process all chunks in a single run
        file_mode = 'w'
        annotations = all_annotations
        print(f"Processing all {len(annotations)} annotations in a single run...")
    elif args.num_chunks > 1:
        # Multi-chunk processing (one chunk at a time)
        if args.chunk_idx == 0:
            # First chunk - create new file
            file_mode = 'w'
            subarrays = [[] for _ in range(args.num_chunks)]
            for i in range(args.num_chunks):
                subarrays[i] = all_annotations[i::args.num_chunks]
            annotations = subarrays[0]
        else:
            # Subsequent chunks - append to existing file
            file_mode = 'a'
            subarrays = [[] for _ in range(args.num_chunks)]
            for i in range(args.num_chunks):
                subarrays[i] = all_annotations[i::args.num_chunks]
            annotations = subarrays[min(args.chunk_idx, len(subarrays)-1)]
    else:
        # Single chunk processing
        file_mode = 'w'
        annotations = all_annotations

    # In this script, we only support direct answering (no zoom search)
    results_file = open(args.answers_file, file_mode)
    
    for annotation in tqdm(annotations):
        # resposne = get_direct_response(model, processor, annotation, question_messages, option_messages, image_folder)
        response, loss_list = multiple_choices_inference(model, processor, tokenizer, annotation, messages, image_folder)
        annotation['output'] = response
        annotation['loss'] = loss_list
        results_file.write(json.dumps(annotation) + "\n")
    results_file.close()

    print(f"Wrote {len(annotations)} answers to {args.answers_file}")
    
    # If this is the last chunk, provide summary
    if args.num_chunks > 1 and args.chunk_idx == args.num_chunks - 1:
        print(f"Completed processing all {args.num_chunks} chunks. Total results written to {args.answers_file}")
    elif args.process_all_chunks:
        print(f"Completed processing all annotations in a single run. Total results written to {args.answers_file}")
