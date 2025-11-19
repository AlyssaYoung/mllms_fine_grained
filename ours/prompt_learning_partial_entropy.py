import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention,apply_multimodal_rotary_pos_emb, repeat_kv
from qwen_vl_utils import process_vision_info
import torch
import pdb 
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
from typing import Dict, List, Optional, Sequence, Tuple
from transformers.cache_utils import Cache
import math 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from qwen_2_5_vl_utils import (
    get_grid_shape, build_mask_from_bbox, compute_activation_loss_qwen, load_qwen_model
)
import copy
from torch.nn import CrossEntropyLoss
from collections.abc import Callable

def get_visual_token_indices(input_ids: torch.Tensor, tokenizer: AutoTokenizer) -> List[int]:
    # DEFAULT_IMAGE_TOKEN = 151655
    DEFAULT_IMAGE_ST = 151652
    DEFAULT_IMAGE_ED = 151653
    ids = input_ids[0].tolist()  # [S_text]
    try: 
        start = ids.index(DEFAULT_IMAGE_ST) + 1
        end = ids.index(DEFAULT_IMAGE_ED)
    except ValueError:
        return []
    return list(range(start, end))

def update_visual_prompt(
    visual_prompt: torch.nn.Parameter,
    grad: torch.Tensor,
    state: Dict[str, torch.Tensor],
    hyperparams: Dict[str, float],
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-3,
) -> None:
    state['m'] = beta1 * state['m'] + (1 - beta1) * grad
    state['s'] = beta2 * state['s'] + (1 - beta2) * grad.pow(2)
    m_hat = state['m'] / (1 - beta1 ** hyperparams['t'])
    s_hat = state['s'] / (1 - beta2 ** hyperparams['t'])

    visual_prompt.data = visual_prompt.data - hyperparams['lr'] * m_hat / (torch.sqrt(s_hat) + eps)
    hyperparams['t'] += 1

def add_prompt_hook(visual_prompt):
    def _hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
        return out + visual_prompt.to(out.device)
    return _hook

def compute_option_loss(
        model, 
        question_logits, 
        past_key_values, 
        option_prompt, 
        procesor, 
        question_inputs, 
        image_inputs
    ):
    option_inputs = processor(text=[option_prompt], images=image_inputs, return_tensors="pt")
    option_inputs = option_inputs.to(model.device)
    q_len = question_logits.shape[1]
    option_ids = option_inputs.input_ids[:, q_len:]
    labels = option_ids.reshape(-1)

    option_attn_mask = torch.ones_like(
        option_ids, dtype=question_inputs.attention_mask.dtype, device=model.device
    )
    full_attn_mask = torch.cat([
        question_inputs.attention_mask, option_attn_mask
    ], dim=1)

    past_len = q_len
    cache_position = torch.arange(
        past_len, 
        past_len + option_ids.shape[1], 
        device=option_ids.device, 
        dtype=torch.long
    )

    output = model(
        input_ids=option_ids,
        attention_mask=full_attn_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
        pixel_values=None,
        image_grid_thw=question_inputs.image_grid_thw,
    )

    logits = torch.cat(
        [question_logits[:, -1:], output.logits[:, :-1]\
    ], dim=1).reshape(-1, model.config.vocab_size)

    loss_fn = CrossEntropyLoss()
    return loss_fn(logits.float(), labels)

def build_option_prompt(shared_messages, option, processor):
    option_messages = shared_messages + [{"role": "assistant", "content": option}]
    option_prompt = processor.apply_chat_template(
        option_messages, tokenize=False, add_generation_prompt=False
    )
    return option_prompt

def main(
    args: argparse.Namespace,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    annotation: Dict,
    messages_fn,
    image_folder: str,
) -> int:
    input_image = annotation["input_image"]
    image_path = os.path.join(image_folder, input_image)
    pil_image = Image.open(image_path).convert("RGB")
    question = annotation["question"]
    shared_messages = messages_fn(image_path, question)
    options = annotation.get('options', [])

    question_prompt = processor.apply_chat_template(
        shared_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(shared_messages)
    grid_shape = get_grid_shape(processor, image_inputs)
    # print(f"grid shape: {grid_shape[0] * grid_shape[1]}")

    question_inputs = processor(
        text=[question_prompt], images=image_inputs, return_tensors="pt"
    )
    question_inputs = question_inputs.to(model.device)
    q_len = question_inputs.input_ids.shape[1]

    gt_option = options[0]
    gt_prompt = build_option_prompt(shared_messages, gt_option, processor)
    gt_full_inputs = processor(text=[gt_prompt], images=image_inputs, return_tensors="pt")
    gt_full_inputs = gt_full_inputs.to(model.device)
    gt_o_len = gt_full_inputs.input_ids.shape[1] - q_len
    gt_o_ids = gt_full_inputs.input_ids[:, q_len:]

    v_token_indices = get_visual_token_indices(question_inputs.input_ids, tokenizer)
    visual_prompt = torch.nn.Parameter(
        torch.zeros(grid_shape[0] * grid_shape[1], model.config.hidden_size, device=model.device),
        requires_grad=True,
    )
    # Adam state.
    state = {
        "m": torch.zeros_like(visual_prompt),
        "s": torch.zeros_like(visual_prompt),
    }
    hyperparams = {"lr": args.lr, "t": 1}
    # Install hooks: one for the visual prompt injection and another for
    # capturing attention weights from specified layers.
    prompt_handle = model.visual.register_forward_hook(add_prompt_hook(visual_prompt))
    v_token_indices_tensor = torch.tensor(v_token_indices, device=model.device)

    # === T-1 steps of prompt updates ===
    with torch.set_grad_enabled(True):
        for t in range(args.T - 1):
            output = model(
                input_ids=gt_full_inputs.input_ids,
                attention_mask=gt_full_inputs.attention_mask,
                pixel_values=gt_full_inputs.pixel_values,
                image_grid_thw=gt_full_inputs.image_grid_thw,
                use_cache=False,
            )
            logits = output.logits
            option_logits = logits[:, q_len-1: q_len+gt_o_len-1].reshape(-1, model.config.vocab_size)
            loss = F.cross_entropy(option_logits.float(), gt_o_ids.reshape(-1))
            grad = torch.autograd.grad(loss, visual_prompt, retain_graph=False)[0]
            update_visual_prompt(visual_prompt, grad, state, hyperparams, args.beta1, args.beta2, args.eps)

    # === Final step: evaluate all options ===
    with torch.no_grad():
        # === Get initial question logits ===
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
        rope_deltas = output.rope_deltas

        model.rope_deltas = rope_deltas.to(model.device)
        initial_past_key_values = copy.deepcopy(past_key_values)
        losses = []
        for option in options:
            past_key_values = copy.deepcopy(initial_past_key_values)
            option_prompt = build_option_prompt(shared_messages, option, processor)
            loss = compute_option_loss(
                model, question_logits, past_key_values, 
                option_prompt, processor, question_inputs, image_inputs
            )
            losses.append(loss)
    option_chosen = torch.stack(losses).argmin().item()
    del past_key_values, initial_past_key_values, visual_prompt
    torch.cuda.empty_cache()

    prompt_handle.remove()
    return option_chosen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run qwen2.5_vl inference with ZoomEye-compatible outputs")
    parser.add_argument("--model_path", type=str, default="/data1/pinci/ckpt/huggingface/Qwen2.5-VL-3B-Instruct", help="Path to LLaVA model checkpoint")
    parser.add_argument("--answers_file", type=str, default=None, help="Path to output answers .jsonl file")
    parser.add_argument("--annotation_path", type=str, default="/data1/pinci/datasets/zoom_eye_data", help="Path to dataset root (contains benchmark folders)")
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
    parser.add_argument("--max_pixels", type=int, default=5500000, help="Max pixels")
    # parser.add_argument("--dinov3_feat_path", type=str, required=True, help="Path to dinov3 features")

    args = parser.parse_args()
    args.beta1, args.beta2, args.eps = 0.9, 0.999, 1e-3
    args.T= 5 # 2
    args.lr = 0.02
    args.alpha = 400
    
    model, processor, tokenizer = load_qwen_model(args.model_path, max_pixels=args.max_pixels)
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

    for p in model.parameters():
        p.requires_grad = False

    for annotation in tqdm(annotations):
        response = main(args, model, processor, tokenizer, annotation, messages, image_folder)
        print(response)
        annotation['output'] = response
        results_file.write(json.dumps(annotation) + "\n")
    results_file.close()

    print(f"Wrote {len(annotations)} answers to {args.answers_file}")

    if args.num_chunks > 1 and args.chunk_idx == args.num_chunks - 1:
        print(f"Completed processing all {args.num_chunks} chunks. Total results written to {args.answers_file}")
    elif args.process_all_chunks:
        print(f"Completed processing all annotations in a single run. Total results written to {args.answers_file}")


    # image_path = "/data1/pinci/datasets/zoom_eye_data/vstar/relative_position/sa_6183.jpg"
    # question = "Is the motorcycle on the left or right side of the dog?"
    # target_object = ["dog", "motorcycle"]
    # bbox_list = [[1455,1302,117,77],[684,945,150,110]]