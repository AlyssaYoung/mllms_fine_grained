# main.py (clean and modular version)

import argparse
import json
import os
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, "/data1/pinci/code/mllms_fine_grained")

from config import DefaultConfig
from prompt_builder import build_question_prompt, build_option_prompt, prepare_inputs
from vision_utils import load_image_and_vision_inputs
from prompt_optimizer import optimize_visual_prompt
from option_evaluator import evaluate_all_options
from hook_utils import visual_prompt_hook
from qwen_2_5_vl_utils import load_qwen_model


def run_single_example(
    args,
    model,
    processor,
    tokenizer,
    annotation,
    messages_fn,
    image_folder,
    cfg,
):
    """Run a single VQA multiple choice example with prompt optimization."""
    
    # === 1. 读取图像 + vision inputs  ===
    image_path = os.path.join(image_folder, annotation["input_image"])

    # === 2. 构造 question prompt ===
    question = annotation["question"]
    shared_messages = messages_fn(image_path, question)
    question_prompt = build_question_prompt(shared_messages, processor)
    image_inputs, grid_shape = load_image_and_vision_inputs(shared_messages, processor)

    # === 3. 生成 question_inputs（含 pixel_values & text tokens）===
    question_inputs = prepare_inputs(processor, question_prompt, image_inputs, device=model.device)
    q_len = question_inputs.input_ids.shape[1]

    # === 4. 初始化视觉 prompt 参数（trainable） ===
    visual_prompt = torch.nn.Parameter(
        torch.zeros(grid_shape[0] * grid_shape[1], model.config.hidden_size, device=model.device)
    )

    # === 5. 安全注册 Hook，将 visual_prompt 注入 VisionTower ===
    with visual_prompt_hook(model, visual_prompt):

        # === 6. T-1 次更新 visual prompt ===
        optimize_visual_prompt(
            model=model,
            processor=processor,
            annotation=annotation,
            shared_messages=shared_messages,
            image_inputs=image_inputs,
            question_inputs=question_inputs,
            visual_prompt=visual_prompt,
            cfg=cfg,
        )

        # === 7. 最后一步：评估所有选项（含 KV cache） ===
        option_list = annotation.get("options", [])
        chosen_idx = evaluate_all_options(
            model=model,
            processor=processor,
            question_inputs=question_inputs,
            shared_messages=shared_messages,
            options=option_list,
            image_inputs=image_inputs,
        )

    return chosen_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--answer_tag", type=str, required=True)
    parser.add_argument("--answers_file", type=str, default=None, help="Path to output answers .jsonl file")
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--process_all_chunks", action="store_true", help="Process all chunks in a single run (ignores chunk_idx)")
    parser.add_argument("--direct_answer", action="store_true")
    parser.add_argument("--max_pixels", type=int, default=5500000)
    args = parser.parse_args()

    # === Load model ===
    model, processor, tokenizer = load_qwen_model(
        args.model_path, max_pixels=args.max_pixels
    )
    for p in model.parameters():
        p.requires_grad = False

    cfg = DefaultConfig()

    if args.answers_file is None:   
        answers_dir = f"eval/answers/{args.benchmark}"
        answers_dir = os.path.join(answers_dir, os.path.basename(args.model_path))
        os.makedirs(answers_dir, exist_ok=True)
        args.answers_file = os.path.join(answers_dir, f"{args.answer_tag}.jsonl")
        print(args.answers_file)

    annotation_file = os.path.join(
        args.annotation_path,
        f"{args.benchmark}/annotation_{args.benchmark}.json"
    )

    with open(annotation_file, 'r') as f:
        all_annotations = json.load(f)

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
    results_file = open(args.answers_file, file_mode)

    # Template for building messages
    messages_fn = lambda img, q: [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}
    ]
    image_folder = os.path.join(args.annotation_path, args.benchmark)

    # === Run Each Example ===
    for ann in tqdm(annotations):
        idx = run_single_example(
            args, model, processor, tokenizer,
            ann, messages_fn, image_folder, cfg
        )
        ann["output"] = idx
        results_file.write(json.dumps(ann) + "\n")

    results_file.close()
    print(f"Done. Results written to {args.answers_file}")
