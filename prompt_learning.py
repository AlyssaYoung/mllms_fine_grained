import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, Qwen2_5_VLFlashAttention2, apply_multimodal_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import _flash_attention_forward
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

class HookedQwenAttention(Qwen2_5_VLAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` shape wrong: {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # if output_attentions:
        #     print(f"[Hook] attn_weights shape = {attn_weights.shape}")

        return attn_output, attn_weights, past_key_value

def visualize_attn_map(pil_image_resized: Image.Image, visual_attn_up: torch.Tensor, cmap='jet', alpha=0.5, save_path="vis_attn.jpg"):
    # 转为 numpy 格式
    attn_map = visual_attn_up.float().numpy()
    
    # 归一化到 0~1 之间
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # resize 原图为 numpy 格式
    image_np = np.array(pil_image_resized).astype(np.float32) / 255.0

    # 使用 matplotlib colormap 映射 attention 到 RGB
    cmap_fn = plt.get_cmap(cmap)
    attn_color = cmap_fn(attn_map)[:, :, :3]  # 去掉 alpha 通道

    # 融合图像和 attention map
    overlay = (1 - alpha) * image_np + alpha * attn_color
    overlay = np.clip(overlay, 0, 1)

    # 显示
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Attention Overlay")
    plt.savefig(save_path)

def merge_bboxes(bbox1, bbox2):
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )

def union_all_bboxes(bboxes):
    if len(bboxes) == 0:
        return None
    ret = [int(bboxes[0][0]), int(bboxes[0][1]), int(bboxes[0][0]+bboxes[0][2]), int(bboxes[0][1]+bboxes[0][3])]
    for bbox in bboxes[1:]:
        new_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]
        ret = merge_bboxes(ret, new_bbox)
    return ret

def get_range_from_bbox_list(bbox_list):
    bbox_cnts = len(bbox_list)
    if bbox_cnts == 1:
        x_min, y_min, x_max, y_max = int(bbox_list[0][0]), int(bbox_list[0][1]), int(bbox_list[0][0] + bbox_list[0][2]), int(bbox_list[0][1] + bbox_list[0][3])
        return [x_min, y_min, x_max, y_max]
    else:
        merged_bbox = union_all_bboxes(bbox_list)
        return merged_bbox

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

def get_target_object_indices(
    target_object: Sequence[str], input_ids: torch.Tensor, tokenizer: AutoTokenizer
) -> List[List[int]]:
    if isinstance(target_object, str):
        target_object = [target_object]
    ids = input_ids[0].tolist()
    DEFAULT_IMAGE_TOKEN = 151655
    DEFAULT_IMAGE_ST = 151652
    DEFAULT_IMAGE_ED = 151653
    image_tokens = {DEFAULT_IMAGE_ST, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_ED}
    # Build a mapping from the filtered token list back to original indices.
    tokens_no_img: List[int] = []
    idx_map: List[int] = []
    for i, tok in enumerate(ids):
        if tok in image_tokens:
            continue
        tokens_no_img.append(tok)
        idx_map.append(i)
    
    matched: List[List[int]] = []
    for obj in target_object:
        candidate_ids_list: List[List[int]] = []
        ids_no_sp = tokenizer(obj, add_special_tokens=False).input_ids
        ids_sp = tokenizer(" "+obj, add_special_tokens=False).input_ids    
        candidate_ids_list.append(ids_no_sp)
        if ids_sp != ids_no_sp:
            candidate_ids_list.append(ids_sp)
    
        for phrase_ids in candidate_ids_list:
            k = len(phrase_ids)
            for i in range(len(tokens_no_img) - k + 1):
                if tokens_no_img[i:i + k] == phrase_ids:
                    start_idx = idx_map[i]
                    end_idx = idx_map[i + k - 1]
                    matched.append(list(range(start_idx, end_idx + 1)))
    return matched

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


def register_attention_hooks(
    model: Qwen2_5_VLForConditionalGeneration,
    target_layers: Sequence[int],
    attn_maps: List[torch.Tensor],
) -> List[torch.utils.hooks.RemovableHandle]:
    base_model = model.module if hasattr(model, "module") else model
    qwen_model = getattr(base_model, "model", base_model)
    if not hasattr(qwen_model, "layers"):
        raise AttributeError(
            "Unable to locate transformer layers on the provided model."
        )
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for idx in target_layers:
        # Replace the original attention module with our hooked version.
        old_attn = qwen_model.layers[idx].self_attn
        # Construct a new module that mirrors the old one.
        device = next(old_attn.parameters()).device
        dtype = next(old_attn.parameters()).dtype
        hooked_attn = HookedQwenAttention(old_attn.config, layer_idx=old_attn.layer_idx).to(device=device, dtype=dtype)
        hooked_attn.load_state_dict(old_attn.state_dict(), strict=False)
        hooked_attn.layer_idx = old_attn.layer_idx
        qwen_model.layers[idx].self_attn = hooked_attn
        # Register a forward hook to capture the attention weights.
        def _make_hook(layer_idx: int):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: Tuple[torch.Tensor, torch.Tensor, Cache]):
                # ``out`` is (attn_output, attn_weights, past_key_value)
                _, attn_weights, _ = out
                attn_maps.append(attn_weights.squeeze(0))
            return hook
        handle = hooked_attn.register_forward_hook(_make_hook(idx))
        handles.append(handle)
    return handles

def add_prompt_hook(visual_prompt):
    def _hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
        return out + visual_prompt.to(out.device)
    return _hook


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
    options = annotation.get('options', [])
    target_object = annotation.get('target_object', [])
    bbox_list = annotation.get('bbox', [])
    assert len(bbox_list) >= 1
    box_range = get_range_from_bbox_list(bbox_list)

    shared_messages = messages_fn(image_path, question)
    question_prompt = processor.apply_chat_template(
        shared_messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(shared_messages)
    grid_shape = get_grid_shape(processor, image_inputs)
    # print(f"grid shape: {grid_shape[0] * grid_shape[1]}")
    mask = build_mask_from_bbox(
        box_range, image_size=pil_image.size, grid_shape=grid_shape
    ).to(model.device)

    question_inputs = processor(
        text=[question_prompt], images=image_inputs, return_tensors="pt"
    )
    # Move inputs to the primary device used by the model.
    # question_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in question_inputs.items()}
    target_object_indices = get_target_object_indices(target_object, question_inputs.input_ids, tokenizer)
    v_token_indices = get_visual_token_indices(question_inputs.input_ids, tokenizer)
    
    # image_size = question_inputs.image_grid_thw[0]
    # v_token_h, v_token_w = image_size[1] // 2, image_size[2] // 2
    # resized_resolution = (v_token_h * 28, v_token_w * 28)
    # pil_image = Image.open(image_path).convert('RGB')
    # pil_image_resized = pil_image.resize((resized_resolution[1], resized_resolution[0]), resample=Image.BICUBIC)

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
    target_layers = list(range(17, 36, 4))
    attn_maps: List[torch.Tensor] = []
    attn_handles = register_attention_hooks(model, target_layers, attn_maps)
    
    # Precompute a few index tensors on the correct devices.
    all_obj_indices = sorted({idx for group in target_object_indices for idx in group})
    obj_indices_tensor = torch.tensor(all_obj_indices, device=model.device)
    v_token_indices_tensor = torch.tensor(v_token_indices, device=model.device)

    response_T = []
    for t in range(args.T):
        is_last = t == args.T - 1
        attn_maps.clear()
        
        with torch.set_grad_enabled(not is_last):
            if is_last:
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
                model.rope_deltas = output.rope_deltas.to(model.device)
                initial_past_key_values = copy.deepcopy(past_key_values)

                losses: List[torch.Tensor] = []
                for option in options:
                    past_key_values = copy.deepcopy(initial_past_key_values)

                    # Build a prompt that appends the option as assistant output.
                    option_messages = shared_messages + [{"role": "assistant", "content": option}]
                    option_prompt = processor.apply_chat_template(
                        option_messages, tokenize=False, add_generation_prompt=False
                    )

                    option_inputs = processor(text=[option_prompt], images=image_inputs, return_tensors="pt")
                    # option_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in option_inputs.items()}
                    q_len = question_inputs.input_ids.shape[1]
                    option_ids = option_inputs.input_ids[:, q_len:]
                    labels = option_ids.reshape(-1)

                    # Build the full attention mask for incremental decoding.
                    option_attn_mask = torch.ones_like(option_ids, dtype=question_inputs.attention_mask.dtype)
                    full_attn_mask = torch.cat([
                        question_inputs.attention_mask, option_attn_mask
                    ], dim=1)

                    # Compute positions for new tokens.
                    past_len = q_len
                    cache_position = torch.arange(
                        past_len, 
                        past_len + option_ids.shape[1],
                        device=option_ids.device,
                        dtype=torch.long
                    )

                    # Incremental forward: only process new tokens with cache.
                    output_option = model(
                        input_ids=option_ids,
                        attention_mask=full_attn_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=False,
                        return_dict=True,
                        pixel_values=None,                   
                        image_grid_thw=question_inputs.image_grid_thw,
                    )

                    # Align logits: the first token's prediction comes from the question pass.
                    logits = torch.cat(
                        [question_logits[:, -1:], output_option.logits[:, :-1]\
                    ], dim=1).reshape(-1, model.config.vocab_size)

                    loss_fn = CrossEntropyLoss()
                    losses.append(loss_fn(logits.float(), labels))

                    del past_key_values
                del initial_past_key_values

                option_chosen = torch.stack(losses).argmin().item()
                loss_list_out = [l.detach().float().item() for l in losses]
                print(loss_list_out)           

                # generated_ids = model.generate(**question_inputs, max_new_tokens=16)
                # generated_ids_trimmed = [
                #             out_ids[len(in_ids):] for in_ids, out_ids in zip(question_inputs.input_ids, generated_ids)
                #         ]
                # output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # output_T.append(output_text)
                # print(output_text)
                response_T.append(option_chosen)
                break
            else:
                out = model(**question_inputs)
            
            if not attn_maps:
                raise RuntimeError(
                    "No attention maps captured.  Check that target_layers is non‑empty and hooks are working."
                )
            losses_per_map: List[torch.Tensor] = []
            loss_dev = torch.device("cuda:0")
            for attn in attn_maps:
                attn_mean_heads = attn.mean(dim=0)  # (seq_len, seq_len)
                obj_indices_tensor = obj_indices_tensor.to(attn_mean_heads.device)
                v_token_indices_tensor = v_token_indices_tensor.to(attn_mean_heads.device)
                obj_attn = attn_mean_heads[obj_indices_tensor, :].mean(dim=0) # (seq_len,)
                obj2img = obj_attn[v_token_indices_tensor].unsqueeze(0) # (1, n_imgs)
                losses_per_map.append(
                    compute_activation_loss_qwen(obj2img, [mask.to(obj2img.device)]).to(loss_dev)
                )
            # Average losses across hooked layers.
            loss = args.alpha * sum(losses_per_map) / len(losses_per_map)
        
        grad = torch.autograd.grad(loss, visual_prompt, retain_graph=False)[0]
        update_visual_prompt(visual_prompt, grad, state, hyperparams, args.beta1, args.beta2, args.eps)
        torch.cuda.empty_cache()

    prompt_handle.remove()
    for h in attn_handles:
        h.remove()
    
    # pil_image = Image.open(image_path).convert('RGB')
    # pil_image_resized = pil_image.resize((resized_resolution[1], resized_resolution[0]), resample=Image.BICUBIC)
    # visual_attn_up = F.interpolate(
    #     visual_attn_2d.unsqueeze(0).unsqueeze(0),
    #     size=resized_resolution, mode='bicubic', align_corners=False
    # ).squeeze()
    # visualize_attn_map(pil_image_resized, visual_attn_up, save_path=f"all_targets_attn_map.jpg")
    return response_T[0]

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
    # parser.add_argument("--dinov3_feat_path", type=str, required=True, help="Path to dinov3 features")

    args = parser.parse_args()
    args.beta1, args.beta2, args.eps = 0.9, 0.999, 1e-3
    args.T= 5
    args.lr = 0.02
    args.alpha = 400
    
    model, processor, tokenizer = load_qwen_model(args.model_path, multi_gpu=args.multi_gpu)
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