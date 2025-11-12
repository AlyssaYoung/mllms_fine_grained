from test_qwen25_vl import load_qwen_model
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

class HookedQwenAttention(Qwen2_5_VLAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values : Optional[Cache] = None,
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

        if past_key_values  is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values .update(key_states, value_states, self.layer_idx, cache_kwargs)

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

        return attn_output, attn_weights, past_key_values 


def visualize_warp_grid(f_X: torch.Tensor, f_Y: torch.Tensor, save_path: str = None):
    """
    可视化 warp 后的 mesh grid。
    f_X: [W_feat], f_Y: [H_feat]
    """
    W_feat = f_X.shape[0]
    H_feat = f_Y.shape[0]
    grid_x, grid_y = torch.meshgrid(f_X, f_Y, indexing='xy')
    grid_x = grid_x.cpu().numpy()
    grid_y = grid_y.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(0, W_feat):
        ax.plot(grid_x[:, i], grid_y[:, i], color='black', lw=1, alpha=0.6)
    for j in range(0, H_feat):
        ax.plot(grid_x[j, :], grid_y[j, :], color='black', lw=1, alpha=0.6)

    ax.set_aspect('equal')
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def warp_image_by_attention(
    pil_image: Image.Image,
    attn_map: torch.Tensor,
    output_size: Tuple[int, int],
    save_path: str = None,
    grid_save_path: str = None,
) -> Image.Image:
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    
    image_tensor = to_tensor(pil_image).unsqueeze(0)
    _, _, H, W = image_tensor.shape
    H_feat, W_feat = attn_map.shape

    # Normalize attention
    A = attn_map.view(-1)
    A = A / (A.sum() + 1e-6)
    A = A.view(H_feat, W_feat)

    # Marginal attention
    m_x = A.sum(dim=0)
    m_y = A.sum(dim=1)

    # CDFs
    M_x = torch.cumsum(m_x, dim=0) / (m_x.sum() + 1e-6)
    M_y = torch.cumsum(m_y, dim=0) / (m_y.sum() + 1e-6)

    # Inverse CDFs
    j = torch.linspace(1.0 / W_feat, 1.0, W_feat, device=A.device)
    i = torch.linspace(1.0 / H_feat, 1.0, H_feat, device=A.device)

    f_X = torch.zeros_like(j)
    f_Y = torch.zeros_like(i)

    for idx, val in enumerate(j):
        k = (M_x >= val).nonzero(as_tuple=False)[0]
        f_X[idx] = (k.item() / W_feat) * (W - 1)

    for idx, val in enumerate(i):
        k = (M_y >= val).nonzero(as_tuple=False)[0]
        f_Y[idx] = (k.item() / H_feat) * (H - 1)

    # Step: visualize warp grid
    if grid_save_path:
        visualize_warp_grid(f_X, f_Y, save_path=grid_save_path)

    # Build normalized sampling grid
    grid_x, grid_y = torch.meshgrid(f_X, f_Y, indexing='xy')
    # grid_x = grid_x.t()
    # grid_y = grid_y.t()

    grid_x_norm = (grid_x / (W - 1)) * 2.0 - 1.0
    grid_y_norm = (grid_y / (H - 1)) * 2.0 - 1.0
    grid = torch.stack((grid_x_norm, grid_y_norm), dim=2).unsqueeze(0)

    warped = F.grid_sample(
        image_tensor, grid,
        mode='bilinear', padding_mode='border', align_corners=False
    )

    # Resize
    H_out, W_out = output_size
    if (H_feat, W_feat) != (H_out, W_out):
        warped = F.interpolate(warped, size=(H_out, W_out), mode='bicubic', align_corners=False)

    warped_pil = to_pil(warped.squeeze(0).cpu())
    if save_path:
        warped_pil.save(save_path)

    return warped_pil


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

def get_visual_token_indices(input_ids, tokenizer):
    DEFAULT_IMAGE_TOKEN = 151655
    # DEFAULT_IMAGE_ST = 151652
    # DEFAULT_IMAGE_ED = 151653
    # cand_ids = [DEFAULT_IMAGE_ST, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_ED]
    cand_ids = [DEFAULT_IMAGE_TOKEN]

    ids = input_ids[0].tolist()  # [S_text]
    hits = [i for i, tok in enumerate(ids) if tok in cand_ids]
    return hits if len(hits) > 0 else None

def get_target_object_indices(target_object, input_ids, tokenizer):
    if isinstance(target_object, str):
        target_object = [target_object]
    ids = input_ids[0].tolist()
    DEFAULT_IMAGE_TOKEN = 151655
    DEFAULT_IMAGE_ST = 151652
    DEFAULT_IMAGE_ED = 151653
    image_tokens = {DEFAULT_IMAGE_ST, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_ED}

    tokens_no_img, idx_map = [], []
    for orig_idx, tid in enumerate(ids):
        if tid in image_tokens:
            continue
        tokens_no_img.append(tid)
        idx_map.append(orig_idx)
    
    matched_tokens = []

    for obj in target_object:
        candidate_ids_list = []
        ids_no_sp = tokenizer(obj, add_special_tokens=False).input_ids
        ids_sp = tokenizer(" "+obj, add_special_tokens=False).input_ids
            
        candidate_ids_list.append(ids_no_sp)
        if ids_sp != ids_no_sp:
            candidate_ids_list.append(ids_sp)
    
        for phrase_ids in candidate_ids_list:
            k = len(phrase_ids)
            for i in range(len(tokens_no_img) - k + 1):
                if tokens_no_img[i:i + k] == phrase_ids:
                    start_in_full = idx_map[i]
                    end_in_full = idx_map[i + k - 1]
                    # matched_tokens.append({
                    #     "token_ids": phrase_ids,
                    #     "start_index": start_in_full,
                    #     "end_index": end_in_full
                    # })
                    matched_tokens.append(list(range(start_in_full, end_in_full+1)))

    return matched_tokens

def register_hooks(model, target_layer):
    # hidden_states = {}
    handles = []
    attn_maps = []
    # 如果是 DataParallel 或 DDP，真实模型在 model.module 中
    base_model = model.module if hasattr(model, "module") else model

    # Qwen2_5_VLForConditionalGeneration 将 Qwen2_5_VLModel 存在 .model 属性
    qwen_model = getattr(base_model, "model", base_model)

    if not hasattr(qwen_model, "layers"):
        raise AttributeError("无法在模型中找到 qwen_model.layers，请检查模型结构。")
    
    print(f"[Replace] Layer {target_layer}: FlashAttention → Qwen2_5_VLAttention")
    old_attn = qwen_model.layers[target_layer].self_attn
    device = next(old_attn.parameters()).device
    dtype = next(old_attn.parameters()).dtype
    # new_attn = Qwen2_5_VLAttention(old_attn.config).to(device=device, dtype=dtype)
    new_attn = HookedQwenAttention(old_attn.config).to(device=device, dtype=dtype)
    new_attn.load_state_dict(old_attn.state_dict(), strict=False)
    new_attn.layer_idx = old_attn.layer_idx
    qwen_model.layers[target_layer].self_attn = new_attn

    def hook_attn_output(module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is None:
                print(f"[HookWarning] {module.__class__.__name__} 未返回 attn_weights（可能仍是FA内核）")
                return
            try:
                attn_maps.append(attn_weights.squeeze(0).detach().cpu())
                print(f"[Hook] Layer {target_layer} attn_weights shape: {tuple(attn_weights.shape)}")
                # torch.save(attn_weights.detach().cpu(), f"attn_layer{target_layer}.pt")
            except Exception as e:
                print(f"[HookError] Failed to save attn from {module}: {e}")

    target_module = qwen_model.layers[target_layer].self_attn
    h = target_module.register_forward_hook(hook_attn_output)
    handles.append(h)

    print(f"[Hook] Registered on layer {target_layer} ({target_module.__class__.__name__})")

    return attn_maps, handles


def get_mllm_feat(model, processor, tokenizer, image_path, question, target_object):
    messages = lambda img, question: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    shared_messages = messages(image_path, question)
    question_prompt = processor.apply_chat_template(
        shared_messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(shared_messages)
    question_inputs = processor(
        text=[question_prompt],
        images=image_inputs,
        return_tensors="pt",
    )
    question_inputs = question_inputs.to(model.device)
    target_object_indices = get_target_object_indices(target_object, question_inputs.input_ids, tokenizer)
    v_token_indices = get_visual_token_indices(question_inputs.input_ids, tokenizer)

    image_size = question_inputs.image_grid_thw[0]
    v_token_h, v_token_w = image_size[1] // 2, image_size[2] // 2
    resized_resolution = (v_token_h * 28, v_token_w * 28)

    # total_layers = 37
    # target_layers = list(range(4, total_layers, 4))
    # handles = register_hooks(model, target_layers)
    attn_maps, handles = register_hooks(model, target_layer=16)

    with torch.no_grad():
        outputs = model(
            input_ids=question_inputs.input_ids,
            attention_mask=question_inputs.attention_mask,
            pixel_values=question_inputs.pixel_values,
            image_grid_thw=question_inputs.image_grid_thw,
            return_dict=True,
        )

    for h in handles:  # 清除 hook，防止多次 forward 累积
        h.remove()

    v_token_indices_tensor = torch.tensor(v_token_indices, device=attn_maps[0].device)
    pil_image = Image.open(image_path).convert('RGB')
    pil_image_resized = pil_image.resize((resized_resolution[1], resized_resolution[0]), resample=Image.BICUBIC)
    """
    for i, obj_indices in enumerate(target_object_indices):
        obj_indices_tensor = torch.tensor(obj_indices, device=attn_maps[0].device)
        obj_attn = attn_maps[0][:, obj_indices, :] # [len(obj_tokens), n_heads, seq_len] 
        obj_attn = obj_attn.mean(dim=0) # [n_heads, seq_len]

        visual_attn = obj_attn[:, v_token_indices_tensor].mean(dim=0) # [n_img]
        assert visual_attn.shape[0]== v_token_h * v_token_w
        visual_attn_2d = visual_attn.reshape(v_token_h, v_token_w)

        # warped_image = warp_image_by_attention(pil_image_resized, visual_attn_2d, resized_resolution, save_path=f"{target_object[i]}_warped.jpg", grid_save_path=f"{target_object[i]}_warped_grid.jpg")
        # print(visual_attn_2d.shape)
        # visualize attn map on 2D original image
        visual_attn_up = F.interpolate(visual_attn_2d.unsqueeze(0).unsqueeze(0), size=resized_resolution, mode='bicubic', align_corners=False).squeeze()
        warped_image = warp_image_by_attention(pil_image_resized, visual_attn_up, resized_resolution, save_path=f"{target_object[i]}_up_warped.jpg")
        # print(visual_attn_up.shape)
        # visualize_attn_map(pil_image_resized, visual_attn_up, save_path=f"{target_object[i]}.jpg")
    """
    # Step 1: 合并所有 obj_indices
    all_obj_indices = sorted(set(idx for group in target_object_indices for idx in group))
    obj_indices_tensor = torch.tensor(all_obj_indices, device=attn_maps[0].device)

    # Step 2: 提取并平均这些 token 的注意力
    obj_attn = attn_maps[0][:, obj_indices_tensor, :]     # [len(obj_tokens), n_heads, seq_len]
    obj_attn = obj_attn.mean(dim=0)                       # [n_heads, seq_len]

    # Step 3: 提取视觉 token 部分并平均
    visual_attn = obj_attn[:, v_token_indices_tensor].mean(dim=0)  # [n_img_tokens]
    assert visual_attn.shape[0] == v_token_h * v_token_w
    visual_attn_2d = visual_attn.reshape(v_token_h, v_token_w)

    # Step 4: 可视化 attention map（插值回原图大小）
    visual_attn_up = F.interpolate(
        visual_attn_2d.unsqueeze(0).unsqueeze(0),
        size=resized_resolution, mode='bicubic', align_corners=False
    ).squeeze()
    visualize_attn_map(pil_image_resized, visual_attn_up, save_path=f"all_targets_attn_map.jpg")

    # Step 5: warp image
    warped_image = warp_image_by_attention(
        pil_image_resized,
        visual_attn_up,
        resized_resolution,
        save_path=f"all_targets_up_warped.jpg",
    )

if __name__ == "__main__":
    image_path = "/root/dataset/zoom_eye_data/zoom_eye_data/vstar/relative_position/sa_6183.jpg"
    question = "Is the motorcycle on the left or right side of the dog?"
    target_object = ["dog", "motorcycle"]
    model_path = "/root/autodl-tmp/ckpt/Qwen2.5-VL-3B-Instruct"
    model, processor, tokenizer = load_qwen_model(model_path, multi_gpu=True)
    get_mllm_feat(model, processor, tokenizer, image_path, question, target_object)