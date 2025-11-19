import json
import torch
from PIL import Image
import requests
from io import BytesIO
import os
import argparse
from tqdm import tqdm
import warnings
import sys
import random
import numpy as np
import math
import torch.nn.functional as F
warnings.filterwarnings("ignore")

import pdb


import types
from contextlib import contextmanager
import inspect

# Set random seeds for reproducible results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed at the beginning
set_seed(42)

# Add current directory to Python path to import local LLaVA
sys.path.insert(0, "~/code/mllms_fine_grained")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
from llava_runner import (
    load_llava_model,
    multiple_choices_inference as mc_infer,
    free_form_response as ff_resp,
    option_list_response as opt_list_resp,
    multiple_choice_letter_response as mc_letter_resp,
)
from utils import resize_to_patch_grid, majority_tokens_pooling, sliding_window_encode
from torch.nn import CrossEntropyLoss

@torch.inference_mode()
def multiple_choices_inference(args, model, tokenizer, image_processor, annotation, image_folder=None, conv_type="v1", use_qwen=False):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    image = Image.open(input_image).convert('RGB')
    aligned_image = image.resize((args.resized_res, args.resized_res), Image.BICUBIC)

    image_tensor = image_processor.preprocess(
        [aligned_image], return_tensors="pt",
        do_resize=False, do_center_crop=False  # keep exact size
    )["pixel_values"]
    image_tensor = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensor]

    original_encode_images = model.encode_images

    @torch.inference_mode()
    def hooked_encode_images(images):
        vt = model.get_model().get_vision_tower()
        mm_proj = model.get_model().mm_projector
        proj_dtype = next(mm_proj.parameters()).dtype
        device = vt.device

        def _encode_one(img):
            B, C, H, W = img.shape
            # print(f"img shape: {img.shape}")
            out = vt.vision_tower(
                img.to(device=vt.device, dtype=vt.dtype),
                output_hidden_states=True, 
                interpolate_pos_encoding=True
            )

            feats = vt.feature_select(out).to(dtype=proj_dtype)  # expect [N, K*K, D]
            # print(f"feats shape: {feats.shape}")
            projected = mm_proj(feats)
            return projected
        
        if isinstance(images, list):
            outs = [_encode_one(image) for image in images]
            return torch.cat(outs, dim=0)
        else:
            return _encode_one(images)
    model.encode_images = hooked_encode_images

    conv = conv_templates[conv_type].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    question_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    # test system prompt length:
    # positions = (question_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
    # print(positions)
    # batch_idx, seq_idx = positions[0].tolist()
    # SYS_TOKEN_LEN = int(seq_idx)
    # print(f"SYS_TOKEN_LEN: {SYS_TOKEN_LEN}") # 35 for llava-v1.5-7b
    # pdb.set_trace()
    
    # Track token lengths
    question_token_length = question_input_ids.shape[1]
    # print(f"Question token length: {question_token_length}")

    try:
        output_question = model(
            question_input_ids,
            use_cache=True,
            images=image_tensor,
        )
    finally:
        model.encode_images = original_encode_images
    question_logits = output_question.logits
    question_past_key_values = output_question.past_key_values

    loss_list = []
    max_token_length = question_token_length  # Initialize with question length
    for option in options:
        full_conv = conv_templates[conv_type].copy()
        full_conv.append_message(full_conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
        full_conv.append_message(full_conv.roles[1], option)
        full_prompt = full_conv.get_prompt()
        full_input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]
        
        # Track token lengths for each option
        full_token_length = full_input_ids.shape[1]
        option_token_length = option_answer_input_ids.shape[1]
        # print(f"Option {option}: Full token length: {full_token_length}, Option-only token length: {option_token_length}")
        
        # Update maximum token length
        max_token_length = max(max_token_length, full_token_length)

        output_option = model(
            input_ids=option_answer_input_ids,
            use_cache=True,
            attention_mask=torch.ones(1, question_logits.shape[1]+option_answer_input_ids.shape[1], device=full_input_ids.device),
            past_key_values=question_past_key_values
        )

        logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)
        loss_fct = CrossEntropyLoss()
        logits = logits.view(-1, model.config.vocab_size)
        labels = option_answer_input_ids.view(-1)
        loss = loss_fct(logits, labels)
        loss_list.append(loss)
    
    option_chosen = torch.stack(loss_list).argmin()
    # Convert per-option losses to Python floats for JSON serialization
    per_option_losses = [float(l.detach().cpu()) for l in loss_list]
    
    # Print the longest token length found
    # print(f"Longest token length in forward process: {max_token_length}")
    
    return option_chosen.detach().cpu().item(), per_option_losses

def format_question(question, option_str):
    return question + "\n" + option_str + "Answer the option letter directly."

def format_question_multichoice(question, options):
    ret = question
    for o in options:
        ret += '\n'
        ret += o
    ret += '\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:'
    return ret

@torch.inference_mode()
def get_response(model, tokenizer, image_processor, annotation, image_folder=None, conv_type="v1", use_qwen=False):
    input_image = annotation['input_image']
    if image_folder is not None:
        input_image = os.path.join(image_folder, input_image)
    question = annotation['question']
    options = annotation.get('options', None)

    image = Image.open(input_image).convert('RGB')
    aligned_image, _, _ = resize_to_patch_grid(image) # align image to the dinov3 input
    answer_type = annotation.get('answer_type', 'free_form')

    if answer_type == "logits_match":
        return mc_infer(model, tokenizer, image_processor, aligned_image, question, options, conv_type, use_qwen)
    elif answer_type == "free_form":
        return ff_resp(model, tokenizer, image_processor, aligned_image, question, conv_type, use_qwen)
    elif answer_type == "option_list":
        return opt_list_resp(model, tokenizer, image_processor, aligned_image, question, options, conv_type, use_qwen)
    elif answer_type == "Multiple Choice":
        return mc_letter_resp(model, tokenizer, image_processor, aligned_image, question, options, conv_type, use_qwen)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaVA-v1.5 inference with ZoomEye-compatible outputs")
    parser.add_argument("--model_path", type=str, default="/data1/pinci/ckpt/huggingface/llava-v1.5-7b", help="Path to LLaVA model checkpoint")
    parser.add_argument("--answers_file", type=str, default=None, help="Path to output answers .jsonl file")
    parser.add_argument("--annotation_path", type=str, default="/data1/pinci/datasets/zoom_eye_data", help="Path to dataset root (contains benchmark folders)")
    parser.add_argument("--benchmark", type=str, choices=["vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld"], default="vstar")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--process_all_chunks", action="store_true", help="Process all chunks in a single run (ignores chunk_idx)")
    parser.add_argument("--direct_answer", action="store_true")
    parser.add_argument("--conv_type", type=str, default="v1", help="Conversation template type")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (ignored if --multi_gpu)")
    parser.add_argument("--vision_tower_path", type=str, default=None, help="Path to vision tower weights (for llava-ov-7b)")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU inference (device_map=auto)")
    parser.add_argument("--answer_tag", type=str, required=True, help="Answer tag")
    parser.add_argument("--resized_res", type=int, default=336, help="Resized resolution")
    # parser.add_argument("--dinov3_feat_path", type=str, required=True, help="Path to dinov3 features")

    args = parser.parse_args()
    # Load model (supports both v1.5 and OneVision backends)
    model, tokenizer, image_processor, default_conv_type, use_qwen = load_llava_model(
        model_path=args.model_path,
        vision_tower_path=args.vision_tower_path,
        device=args.device,
        multi_gpu=args.multi_gpu,
        attn_implementation="flash_attention_2",  # Temporarily disabled flash-attn
        torch_dtype=torch.float16,
        padding_side="left",
    )

    # dinov3_feat = torch.load(args.dinov3_feat_path, map_location='cpu')

    if args.conv_type in [None, "auto"]:
        args.conv_type = default_conv_type

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
    # for debug for ntk rope position embedding
    for annotation in tqdm(annotations):
        #dinov3_feat_size = dinov3_feat[annotation['input_image']]['resized_resolution']
        response = multiple_choices_inference(
            args=args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            annotation=annotation,
            image_folder=image_folder,
        )
        if isinstance(response, tuple):
            output_choice, per_option_losses = response
        else:
            output_choice, per_option_losses = response, None

        annotation['output'] = output_choice
        if per_option_losses is not None:
            annotation['per_option_losses'] = per_option_losses
        results_file.write(json.dumps(annotation) + "\n")
    results_file.close()
    
    print(f"Wrote {len(annotations)} answers to {args.answers_file}")
    
    # If this is the last chunk, provide summary
    if args.num_chunks > 1 and args.chunk_idx == args.num_chunks - 1:
        print(f"Completed processing all {args.num_chunks} chunks. Total results written to {args.answers_file}")
    elif args.process_all_chunks:
        print(f"Completed processing all annotations in a single run. Total results written to {args.answers_file}")
