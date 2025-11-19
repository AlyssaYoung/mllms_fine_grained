# vision_utils.py

from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from qwen_2_5_vl_utils import get_grid_shape


def load_image_and_vision_inputs(shared_messages, processor):
    """
    Load image and produce vision tower inputs (pixel_values + grid).
    """
    image_inputs, _ = process_vision_info(shared_messages)
    # Ensure grid_thw inside image_inputs
    grid_shape = get_grid_shape(processor, image_inputs)

    return image_inputs, grid_shape


def get_visual_token_indices(input_ids, tokenizer):
    """
    (Optional) Extract visual token indices based on special start/end IDs.
    Only works if your tokenizer is using DEFAULT_IMAGE_ST/ED scheme.
    """
    DEFAULT_IMAGE_ST = 151652
    DEFAULT_IMAGE_ED = 151653

    ids = input_ids[0].tolist()
    try:
        start = ids.index(DEFAULT_IMAGE_ST) + 1
        end = ids.index(DEFAULT_IMAGE_ED)
    except ValueError:
        return []
    return list(range(start, end))
