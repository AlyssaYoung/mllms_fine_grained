import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, apply_multimodal_rotary_pos_emb, repeat_kv
from qwen_vl_utils import process_vision_info
import torch
import pdb 
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from qwen_2_5_vl_utils import get_grid_shape, build_mask_from_bbox
from typing import Dict, List, Optional, Sequence, Tuple

def visualize_attn_grid(pil_image_resized: Image.Image,
                        attn_map_list: list,    # list of 36 attention maps, each shape [H, W]
                        grid_size=(6, 6),
                        cmap='jet',
                        alpha=0.5,
                        save_path="attn_grid.jpg"):
    """
    Visualize 36 attention maps overlayed on original image, arranged in grid
    """
    assert len(attn_map_list) == grid_size[0] * grid_size[1], "Expected 36 maps for 6x6 grid"

    # Resize original image to numpy
    image_np = np.array(pil_image_resized).astype(np.float32) / 255.0
    H, W = image_np.shape[:2]

    fig, axes = plt.subplots(*grid_size, figsize=(grid_size[1]*3, grid_size[0]*3))
    for i, ax in enumerate(axes.flat):
        attn_map = attn_map_list[i].float().cpu().numpy()

        # Normalize to [0,1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Colorize attention map
        cmap_fn = plt.get_cmap(cmap)
        attn_color = cmap_fn(attn_map)[..., :3]  # drop alpha

        # Resize attention color to match image
        if attn_color.shape[:2] != (H, W):
            attn_color = np.array(Image.fromarray((attn_color*255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0

        # Overlay
        overlay = (1 - alpha) * image_np + alpha * attn_color
        overlay = np.clip(overlay, 0, 1)

        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"Layer {i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_key_object_attn_curve(
    key_object_attn_percents: list,
    color: str = "blue",
    title: str = "Visual Attention to Key Objects across Layers",
    save_path: str = "key_object_attn_curve.jpg"
):
    plt.figure(figsize=(10, 4))
    x = list(range(len(key_object_attn_percents)))
    y = [v * 100 if v < 1.5 else v for v in key_object_attn_percents]  # auto scale to %

    plt.plot(x, y, color=color, linewidth=2, marker='o', markersize=4)
    plt.fill_between(x, y, color=color, alpha=0.2)

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Attention (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(range(0, len(y), 2))
    plt.ylim(0, max(y) * 1.2 if y else 1)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compute_key_object_attn_percent(
    attn_map: torch.Tensor,                   # [H, seq_len, seq_len]
    key_text_token_indices: torch.Tensor,         # [N_t]
    image_token_indices: torch.Tensor,        # [N_v]
    key_object_mask: torch.Tensor    # [N_v], 0/1
) -> torch.Tensor:
    H, _, seq_len = attn_map.shape
    assert key_object_mask.shape == image_token_indices.shape
    
    # average over key text query positions, (H, seq_len)
    attn_selected = attn_map[:, key_text_token_indices, :]
    if attn_selected.dim() == 2:  # i.e., [H, seq_len]
        attn_selected = attn_selected.unsqueeze(1)  # → [H, 1, seq_len]
    attn_to_all = attn_selected.detach().cpu().mean(dim=1)  # → [H, seq_len]
    
    image_token_indices = image_token_indices.detach().cpu()
    attn_to_image_tokens = attn_to_all[:, image_token_indices] # (H, N_v)
    key_object_mask = key_object_mask.detach().cpu()
    attn_to_key = (attn_to_image_tokens * key_object_mask.unsqueeze(0)).sum(dim=-1)  # (H,)
    attn_to_all_img = attn_to_image_tokens.sum(dim=-1)  # (H,)

    eps = 1e-8
    key_object_to_image_percent = attn_to_key.mean() / (attn_to_all_img.mean() + eps)

    visual_to_all_percent = attn_to_all_img.mean() / (attn_to_all.sum(dim=-1).mean() + eps)

    return key_object_to_image_percent, visual_to_all_percent


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

def load_qwen_model(model_path, max_pixels = 12845056):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, # torch_dtype=torch.bfloat16,  # Use float16 instead of auto
                device_map="auto",
                attn_implementation="eager"  # Enable flash attention for memory efficiency
            )
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    max_pixels = max_pixels
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, processor, tokenizer

def main(model, processor, tokenizer, image_path, question, target_object, bbox_list, vis_path, pt_save_path):
    pil_image = Image.open(image_path).convert("RGB")
    messages = lambda img, question: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    shared_messages = messages(image_path, question)
    question_prompt = processor.apply_chat_template(
        shared_messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(shared_messages)
    grid_shape = get_grid_shape(processor, image_inputs)
    print(f"grid shape: {grid_shape}")
    resized_resolution = (grid_shape[0] * 28, grid_shape[1] * 28)
    pil_image_resized = pil_image.resize((resized_resolution[1], resized_resolution[0]), resample=Image.BICUBIC)

    box_range = get_range_from_bbox_list(bbox_list)
    mask = build_mask_from_bbox(
        box_range, image_size=pil_image.size, grid_shape=grid_shape
    ).to(model.device)
    # print(f"mask shape: {mask.shape}")

    question_inputs = processor(
        text=[question_prompt],
        images=image_inputs,
        return_tensors="pt",
    )
    question_inputs = question_inputs.to(model.device)
    target_object_indices = get_target_object_indices(target_object, question_inputs.input_ids, tokenizer)
    all_obj_indices = sorted({idx for group in target_object_indices for idx in group})
    obj_indices_tensor = torch.tensor(all_obj_indices, device=model.device)
    v_token_indices = get_visual_token_indices(question_inputs.input_ids, tokenizer)
    v_token_indices_tensor = torch.tensor(v_token_indices, device=model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=question_inputs.input_ids,
            attention_mask=question_inputs.attention_mask,
            pixel_values=question_inputs.pixel_values,
            image_grid_thw=question_inputs.image_grid_thw,
            return_dict=True,
            output_attentions=True,
        )
    attn_maps = outputs.attentions
    num_layers = len(attn_maps)

    obj2img_attn_maps = []
    key_object_attn_percents = []
    visual_attn_percents = []
    os.makedirs(os.path.join(vis_path, "attn_map_grid"), exist_ok=True)
    os.makedirs(os.path.join(vis_path, "obj_percent_curve"), exist_ok=True)
    os.makedirs(os.path.join(vis_path, "visual_percent_curve"), exist_ok=True)
    parts = os.path.normpath(image_path).split(os.sep)
    base_name = f"{parts[-2]}_{parts[-1]}"[:-4]

    print(f"base_name: {base_name}")

    for layer_idx in range(num_layers):
        # attn_mean_heads = attn_maps[layer_idx].mean(dim=1).squeeze(0) # (seq_len, seq_len)
        # obj_indices_tensor = obj_indices_tensor.to(attn_mean_heads.device) 
        # v_token_indices_tensor = v_token_indices_tensor.to(attn_mean_heads.device)
        # obj_attn = attn_mean_heads[obj_indices_tensor, :].mean(dim=0) # (seq_len,)
        # obj2img = obj_attn[v_token_indices_tensor].detach().cpu()
        # obj2img_2d = obj2img.reshape(grid_shape[0], grid_shape[1])
        # visual_attn_2d = F.interpolate(obj2img_2d.unsqueeze(0).unsqueeze(0), size=resized_resolution, mode='bicubic', align_corners=False).squeeze()
        # obj2img_attn_maps.append(visual_attn_2d)
        key_object_to_image_percent, visual_to_all_percent = compute_key_object_attn_percent(attn_maps[layer_idx].squeeze(0), obj_indices_tensor.squeeze(0), v_token_indices_tensor.squeeze(0), mask.flatten())
        key_object_attn_percents.append(key_object_to_image_percent.item())
        visual_attn_percents.append(visual_to_all_percent.item())
    
    # visualize_attn_grid(
    #     pil_image_resized,
    #     obj2img_attn_maps,
    #     grid_size=(6, 6),
    #     save_path=f"{vis_path}/attn_map_grid/{base_name}.jpg"
    # )
    # print(key_object_attn_percents)
    # plot_key_object_attn_curve(
    #     key_object_attn_percents,
    #     title="Visual Attention to Key Objects across Layers",
    #     save_path=f"{vis_path}/obj_percent_curve/{base_name}.jpg"
    # )
    plot_key_object_attn_curve(
        visual_attn_percents,
        title="Visual Attention to All Tokens across Layers",
        save_path=f"{vis_path}/visual_percent_curve/{base_name}.jpg"
    )

    # save pt file
    # attn_layer_data = {
    #     "image_path": image_path,
    #     "attn_maps": [a.cpu() for a in attn_maps],  # keep full [1, H, S, S]
    #     "obj_token_indices": obj_indices_tensor.cpu(),
    #     "v_token_indices": v_token_indices_tensor.cpu(),
    #     "mask": mask.flatten().cpu(),  # shape [H*W]
    #     "grid_shape": grid_shape,
    # }
    # os.makedirs(pt_save_path, exist_ok=True)
    # torch.save(attn_layer_data, f"{pt_save_path}/{base_name}.pt")

if __name__ == "__main__":
    # image_path = "/data1/pinci/datasets/zoom_eye_data/vstar/relative_position/sa_6183.jpg"
    # question = "Is the motorcycle on the left or right side of the dog?"
    # target_object = ["dog", "motorcycle"]
    # bbox_list = [[1455,1302,117,77],[684,945,150,110]]
    image_folder = "/data1/pinci/datasets/zoom_eye_data/vstar"
    correct_demo_file = "demo/correct_subset.jsonl"
    incorrect_demo_file = "demo/incorrect_subset.jsonl"

    model_path = "/data1/pinci/ckpt/huggingface/Qwen2.5-VL-3B-Instruct"
    max_pixels = 3600000 # for save GPU memory
    model, processor, tokenizer = load_qwen_model(model_path, max_pixels=max_pixels)

    pt_save_path = "demo/pt_files/correct"
    vis_path = "vis/check_attn_decay/correct"
    with open(correct_demo_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            input_image = item['input_image']
            image_path = os.path.join(image_folder, input_image)
            question = item['question']
            target_object = item['target_object']
            bbox_list = item['bbox']
            main(model, processor, tokenizer, image_path, question, target_object, bbox_list, vis_path, pt_save_path)

    pt_save_path = "demo/pt_files/incorrect"
    vis_path = "vis/check_attn_decay/incorrect"
    with open(incorrect_demo_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            input_image = item['input_image']
            image_path = os.path.join(image_folder, input_image)
            question = item['question']
            target_object = item['target_object']
            bbox_list = item['bbox']
            main(model, processor, tokenizer, image_path, question, target_object, bbox_list, vis_path, pt_save_path)