import torch
from PIL import Image
from llava_runner import load_llava_model
import pdb

import cv2
import os
import json
import numpy as np
from PIL import Image

def visualize_token_retention(
    image,
    retained_tokens: list[int],
    grid_size: int = 28,
    token_size: int = 14,
    save_path: str = 'visualized_token.png'
):

    H, W = 672, 672
    image = np.array(image)
    print(image.shape)  # (672, 672, 3)

    num_rows = 48
    num_cols = 48

    mask = np.zeros_like(image)
    mask_compressed = np.zeros((336, 336, 3), dtype=np.uint8)

    print(len(retained_tokens))  # e.g., 1152
    for idx in retained_tokens:
        row = idx // num_cols
        col = idx % num_cols

        top = row * token_size
        left = col * token_size
        bottom = min((row + 1) * token_size, H)
        right = min((col + 1) * token_size, W)

        # 原图 patch → 原尺寸 mask
        mask[top:bottom, left:right, :] = image[top:bottom, left:right, :]

        # 映射到压缩空间（24x24 grid）
        row_compressed = row // 2
        col_compressed = col // 2
        top_c = row_compressed * token_size
        left_c = col_compressed * token_size
        bottom_c = min((row_compressed + 1) * token_size, 336)
        right_c = min((col_compressed + 1) * token_size, 336)
        mask_compressed[top_c:bottom_c, left_c:right_c, :] = image[top:bottom, left:right, :]

    # Draw red grid on full mask
    for y in range(0, H, grid_size):
        cv2.line(mask, (0, y), (W, y), color=(255, 0, 0), thickness=1)
    for x in range(0, W, grid_size):
        cv2.line(mask, (x, 0), (x, H), color=(255, 0, 0), thickness=1)

    # Draw red grid on compressed
    for x in range(0, 336, 14):
        cv2.line(mask_compressed, (x, 0), (x, 336), color=(255, 0, 0), thickness=1)
    for y in range(0, 336, 14):
        cv2.line(mask_compressed, (0, y), (336, y), color=(255, 0, 0), thickness=1)

    # Resize compressed to same height (672) for vertical concat
    mask_compressed_up = cv2.resize(mask_compressed, (W, H), interpolation=cv2.INTER_NEAREST)

    # Concatenate: top = full resolution, bottom = compressed upsampled
    combined = np.concatenate([mask, mask_compressed_up], axis=0)  # shape: (1344, 672, 3)

    # Save final image
    combined = combined.astype(np.uint8)
    Image.fromarray(combined).save(save_path)


def leave_one_out_mcr(grid_feat: torch.Tensor, eps: float) -> torch.Tensor:
    """
    grid_feat: [B, 4, D]  # batch of 2x2 tokens
    Return: [B, 4] MCR scores for each token
    """
    B, N, D = grid_feat.shape  # N=4
    full_cov = torch.matmul(grid_feat, grid_feat.transpose(1, 2))  # [B, 4, 4]
    I = torch.eye(N, device=grid_feat.device).unsqueeze(0)  # [1, 4, 4]
    M = I + (D / (N * eps**2)) * full_cov
    _, full_logdet = torch.slogdet(M)  # [B]

    leave_one_out = []
    for k in range(N):
        mask = torch.ones(N, dtype=torch.bool, device=grid_feat.device)
        mask[k] = False
        grid_feat_k = grid_feat[:, mask, :]  # [B, 3, D]
        cov_k = torch.matmul(grid_feat_k, grid_feat_k.transpose(1, 2))  # [B, 3, 3]
        I3 = torch.eye(3, device=grid_feat.device).unsqueeze(0)
        M_k = I3 + (D / (3 * eps**2)) * cov_k
        _, logdet_k = torch.slogdet(M_k)  # [B]
        delta = full_logdet - logdet_k  # [B]
        leave_one_out.append(delta.unsqueeze(-1))

    return torch.cat(leave_one_out, dim=-1)  # [B, 4]

def select_tokens_by_mcr(projected_feat_2d: torch.Tensor, eps: float = 0.1) -> list[int]:
    B, H, W, D = projected_feat_2d.shape
    projected_feat_2d = projected_feat_2d.float()

    all_blocks = []
    all_block_positions = []

    for b in range(B):
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                block = projected_feat_2d[b, i:i+2, j:j+2, :]  # shape [2,2,D]
                if block.shape[:2] == (2, 2):
                    block_flat = block.reshape(-1, D)  # [4, D]
                    all_blocks.append(block_flat.unsqueeze(0))  # [1,4,D]
                    all_block_positions.append((b, i, j))

    all_blocks_tensor = torch.cat(all_blocks, dim=0)  # [N_blocks, 4, D]
    mcr_scores = leave_one_out_mcr(all_blocks_tensor, eps)  # [N_blocks, 4]
    best_indices = torch.argmax(mcr_scores, dim=-1)  # [N_blocks]

    retained_token_ids = []
    for idx, (b, i, j) in enumerate(all_block_positions):
        delta_h, delta_w = best_indices[idx] // 2, best_indices[idx] % 2
        token_id = b * (H * W) + (i + delta_h) * W + (j + delta_w)
        retained_token_ids.append(token_id)

    return retained_token_ids


def get_embedding(model, image_tensor):
    vt = model.get_model().get_vision_tower()
    mm_proj = model.get_model().mm_projector
    proj_dtype = next(mm_proj.parameters()).dtype
    device = vt.device
    
    B, C, H, W = image_tensor.shape
    out = vt.vision_tower(
        image_tensor.to(device=vt.device, dtype=vt.dtype),
        output_hidden_states=True, 
        interpolate_pos_encoding=True
    )
    feats = vt.feature_select(out).to(dtype=proj_dtype)  # expect [N, K*K, D]
    projected = mm_proj(feats)
    return projected

if __name__ == "__main__":
    model_path = "/data1/pinci/ckpt/huggingface/llava-v1.5-7b"
    vision_tower_path = "/data1/pinci/ckpt/huggingface/clip-vit-large-patch14-336"
    annotation_path = "/root/dataset/zoom_eye_data/zoom_eye_data"
    benchmark = "vstar"
    annotation_file = os.path.join(annotation_path, f"{benchmark}/annotation_{benchmark}.json")
    with open(annotation_file, 'r') as f:
        all_annotations = json.load(f)

    model, tokenizer, image_processor, default_conv_type, use_qwen = load_llava_model(
        model_path=model_path,
        vision_tower_path=vision_tower_path,
        device="cuda:0",
        multi_gpu=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        padding_side="left",
    )

    for annotation in all_annotations:
        input_image = annotation["input_image"]
        image_path = os.path.join(annotation_path, benchmark, input_image)
        image = Image.open(image_path).convert("RGB")
        aligned_image = image.resize((672, 672), Image.BICUBIC)

        image_tensor = image_processor.preprocess(
            [aligned_image], return_tensors="pt",
            do_resize=False, do_center_crop=False  # keep exact size
        )["pixel_values"]
       
        if isinstance(image_tensor, list):
            projected_feat = [get_embedding(model, image) for image in image_tensor]
            projected_feat = torch.cat(projected_feat, dim=0)
        else:
            projected_feat = get_embedding(model, image_tensor)
        
        print(projected_feat.shape) # [1, 2304, 4096]
        H, W = 48, 48
        B, L, D = projected_feat.shape
        projected_feat_2d = projected_feat.reshape(B, H, W, D)
    
        # Apply MCR-based pruning on 2x2 grids
        retained_tokens = select_tokens_by_mcr(projected_feat_2d, eps=0.1)
        print(f"Number of retained tokens: {len(retained_tokens)}")
        vis_save_path = os.path.join('vis/token_compression', input_image)
        visualize_token_retention(aligned_image, retained_tokens, save_path=vis_save_path)

