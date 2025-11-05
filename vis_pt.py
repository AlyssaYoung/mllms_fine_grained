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
    save_path: str = "key_object_attn_curve.jpg"
):
    plt.figure(figsize=(10, 4))
    x = list(range(len(key_object_attn_percents)))
    y = [v * 100 if v < 1.5 else v for v in key_object_attn_percents]  # auto scale to %

    plt.plot(x, y, color=color, linewidth=2, marker='o', markersize=4)
    plt.fill_between(x, y, color=color, alpha=0.2)

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Key Object Attention (%)", fontsize=12)
    plt.title("Visual Attention to Key Objects across Layers", fontsize=14)
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
    attn_to_all = attn_selected.mean(dim=1)  # → [H, seq_len]
    
    image_token_indices = image_token_indices
    attn_to_image_tokens = attn_to_all[:, image_token_indices] # (H, N_v)
    key_object_mask = key_object_mask
    attn_to_key = (attn_to_image_tokens * key_object_mask.unsqueeze(0)).sum(dim=-1)  # (H,)
    attn_to_all_img = attn_to_image_tokens.sum(dim=-1)  # (H,)

    eps = 1e-8
    percent = attn_to_key.mean() / (attn_to_all_img.mean() + eps)

    return percent


def main(attn_layer_data, vis_path, base_name):
    print(f"{base_name}")
    image_path = attn_layer_data["image_path"]
    attn_maps = attn_layer_data["attn_maps"]
    num_layers = len(attn_maps)
    obj_indices_tensor = attn_layer_data["obj_token_indices"]
    v_token_indices_tensor = attn_layer_data["v_token_indices"]
    mask = attn_layer_data["mask"]
    grid_shape = attn_layer_data["grid_shape"]
    resized_resolution = (grid_shape[0] * 28, grid_shape[1] * 28)
    pil_image = Image.open(image_path).convert("RGB")
    pil_image_resized = pil_image.resize((resized_resolution[1], resized_resolution[0]), resample=Image.BICUBIC)
    num_layers = len(attn_maps)

    obj2img_attn_maps = []
    key_object_attn_percents = []
    os.makedirs(os.path.join(vis_path, "attn_map_grid"), exist_ok=True)
    os.makedirs(os.path.join(vis_path, "curve"), exist_ok=True)

    for layer_idx in range(num_layers):
        attn_mean_heads = attn_maps[layer_idx].mean(dim=1).squeeze(0) # (seq_len, seq_len)
        obj_attn = attn_mean_heads[obj_indices_tensor, :].mean(dim=0) # (seq_len,)
        obj2img = obj_attn[v_token_indices_tensor]
        obj2img_2d = obj2img.reshape(grid_shape[0], grid_shape[1])
        visual_attn_2d = F.interpolate(obj2img_2d.unsqueeze(0).unsqueeze(0), size=resized_resolution, mode='bicubic', align_corners=False).squeeze()
        obj2img_attn_maps.append(visual_attn_2d)
        key_object_attn_percents.append(compute_key_object_attn_percent(attn_maps[layer_idx].squeeze(0), obj_indices_tensor.squeeze(0), v_token_indices_tensor.squeeze(0), mask.flatten()).item())
    
    visualize_attn_grid(
        pil_image_resized,
        obj2img_attn_maps,
        grid_size=(6, 6),
        save_path=f"{vis_path}/attn_map_grid/{base_name}.jpg"
    )
    # print(key_object_attn_percents)
    plot_key_object_attn_curve(
        key_object_attn_percents,
        save_path=f"{vis_path}/curve/{base_name}.jpg"
    )

if __name__ == "__main__":
    correct_pt_file_dir = "demo/pt_files/correct"
    incorrect_pt_file_dir = "demo/pt_files/incorrect"

    for pt_file in os.listdir(correct_pt_file_dir):
        pt_file_path = os.path.join(correct_pt_file_dir, pt_file)
        parts = os.path.normpath(pt_file_path).split(os.sep)
        correct_type = parts[-2]
        base_name = parts[-1][:-3]
        vis_path = f"vis/vis_pt/{correct_type}"
        attn_layer_data = torch.load(pt_file_path)
        main(attn_layer_data, vis_path, base_name)

    for pt_file in os.listdir(incorrect_pt_file_dir):
        pt_file_path = os.path.join(incorrect_pt_file_dir, pt_file)
        parts = os.path.normpath(pt_file_path).split(os.sep)
        correct_type = parts[-2]
        base_name = parts[-1][:-3]
        vis_path = f"vis/vis_pt/{correct_type}"
        attn_layer_data = torch.load(pt_file_path)
        main(attn_layer_data, vis_path, base_name)
    
    