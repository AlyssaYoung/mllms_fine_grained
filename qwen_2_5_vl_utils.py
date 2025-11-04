import numpy as np
from torchvision import transforms
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def load_qwen_model(model_path, multi_gpu):
    # num_gpus = torch.cuda.device_count()

    # max_mem = {i: "23GiB" for i in range(num_gpus)}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, # torch_dtype=torch.bfloat16,  # Use float16 instead of auto
                device_map="auto",
                # max_memory=max_mem,
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

def att_window_from_bbox(bbox, image_size, att_shape, clip=True):
    x1, y1, x2, y2 = bbox
    img_w, img_h    = image_size
    att_h, att_w    = att_shape

    block_w = img_w / att_w
    block_h = img_h / att_h

    x_start = int(np.floor(x1 / block_w))
    y_start = int(np.floor(y1 / block_h))
    x_end   = int(np.ceil (x2 / block_w))
    y_end   = int(np.ceil (y2 / block_h))

    if clip:
        x_start = max(0, min(x_start, att_w))
        y_start = max(0, min(y_start, att_h))
        x_end   = max(0, min(x_end,   att_w))
        y_end   = max(0, min(y_end,   att_h))

    return x_start, y_start, x_end, y_end


def get_grid_shape(processor, image_inputs):
    with torch.no_grad():
        aux = processor.image_processor(images=image_inputs)
    h, w = int(aux["image_grid_thw"][0, 1]/2), int(aux["image_grid_thw"][0, 2]/2)
    # print(f"get_grid_shape: {h}, {w}")
    return h, w

def build_mask_from_bbox(bbox, image_size, grid_shape, device="cpu"):
    h, w = grid_shape
    x0, y0, x1, y1 = att_window_from_bbox(bbox, image_size, (h, w))
    # print(f"att_window_from_bbox: {x0}, {y0}, {x1}, {y1}")
    mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    mask[y0:y1, x0:x1] = 1.0
    return mask

def compute_activation_loss_qwen(rel_map, masks, eps=1e-6):
    if len(masks) == 0:
        return torch.tensor(0., device=rel_map.device)

    B, HW = rel_map.shape
    H, W  = masks[0].shape
    rel_map = rel_map.reshape(B, H, W)

    total_att = rel_map.reshape(B, -1).sum(-1, keepdim=True) + eps

    loss = 0.
    for m in masks:
        act = (rel_map * m).reshape(B, -1).sum(-1) / total_att.squeeze(-1)
        loss += torch.mean((1.0 - act)**2)
    return loss / len(masks)