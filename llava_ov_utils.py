from PIL import Image
from copy import deepcopy
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, divide_to_patches, select_best_resolution, resize_and_pad_image
import torch
from torchvision import transforms
import torch.nn.functional as F
import ast
import re
import math

# dinov3 make_transform
def make_transform(out_w: int, out_h: int):
    return transforms.Compose([
        transforms.Resize((out_h, out_w), antialias=True),  # 注意顺序: (H, W)
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

# zoomeye implementation, quit in our methods
def expand2square(pil_img, background_color):
	width, height = pil_img.size
	if width == height:
		return deepcopy(pil_img), 0, 0
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))
		return result, 0, (width - height) // 2
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))
		return result, (height - width) // 2, 0

# preprocessing for dinov3
def resize_to_patch_grid(pil_img, patch_size= 16):
	width, height = pil_img.size
	def nearest_multiple(n, k):
		n = int(n); k = int(k)
		if n <= 0: 
			return k
		q, r = divmod(n, k)
		down, up = q * k, (q + 1) * k
		return down if (n - down) < (up - n) else up
	resized_w = max(patch_size, nearest_multiple(width,  patch_size))
	resized_h = max(patch_size, nearest_multiple(height, patch_size))
	resized_img = pil_img.resize((resized_w, resized_h), Image.BICUBIC)
	return resized_img, resized_w, resized_h

# transfer clustering label map to one-hot channel map(channel: cluster labels, 
# adaptive_avg_pool2d for region aggergating, token-wise argmax for majority voting,
# finally get token_cluster_map
def majority_tokens_pooling(labels_hw, vt_h, vt_w, num_classes):
    H, W = labels_hw.shape
    device = labels_hw.device
    if num_classes is None: # cluster label classes number
        num_classes = int(labels_hw.max().item()) + 1

    # one-hot: (C, H, W)
    onehot = F.one_hot(labels_hw.view(-1), num_classes=num_classes).T.reshape(num_classes, H, W).float()

    # adaptive_avg_pool2d to (vt_h, vt_w) to get the proportion of each category in each token grid, i.e., majority voting
    pooled = F.adaptive_avg_pool2d(onehot, output_size=(vt_h, vt_w))  # (C, vt_h, vt_w)
    token_cluster_map = pooled.argmax(dim=0).to(torch.long)           # (vt_h, vt_w)
	
    return token_cluster_map

# original implementation from llava.mm_utils.py
def process_images(images, image_processor, model_cfg, show_crop=False):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "highres":
        for image in images:
            image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints, show_crop)
            new_images.append(image)
    elif image_aspect_ratio == "crop_split":
        for image in images:
            image = process_highres_image_crop_split(image, model_cfg, image_processor)
            new_images.append(image)
    elif image_aspect_ratio == "pad":
        for image in images:
            image = image_processor.preprocess(image, do_resize=True, size={"height": 336, "width": 336}, do_center_crop=False, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def get_possible_resolutions(patch_size, grid_pinpoints):
        # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    return possible_resolutions

def process_anyres_image(image, processor, grid_pinpoints, show_crop=False):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    try:
        patch_size = processor.size[0]
    except Exception as e:
        patch_size = processor.size["shortest_edge"]
    assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
    possible_resolutions = get_possible_resolutions(patch_size, grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    # print(555, best_resolution)
    # image_padded = resize_and_pad_image(image, best_resolution)
    image_padded = image.resize(best_resolution, Image.BICUBIC)
    # if show_crop:
    #     print(555, image_padded.size)
    #     image_padded.show()
    #     image_padded.save("/data9/shz/project/llava-ov/LLaVA-NeXT/docs/images/anyres.png")

    patches = divide_to_patches(image_padded, processor.crop_size["height"])
    # for i in range(len(patches)):
    #     print(patches[i].size)
    #     patches[i].save(f"/data9/shz/project/llava-ov/LLaVA-NeXT/docs/images/anyres_{i}.png")

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    # print(666)
    # for x in image_patches:
    #     x.show()
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)

    assert window_size % patch_size == 0
    B, C, H, W = image.shape
    device = device or image.device
    K = window_size // patch_size

    # grid size: compute the global non-overlapping token grid size
    H_token = math.ceil(H / patch_size)
    W_token = math.ceil(W / patch_size)


    # Use dtype from encode_fn output if not specified
    if dtype is None:
        # We'll determine dtype from the first encode_fn call
        dtype = image.dtype
    
    feat_canvas  = torch.zeros(B, H_token, W_token, 0, device=device, dtype=dtype) 
    weight_canvas = torch.zeros(B, H_token, W_token, 1, device=device, dtype=dtype)

    # 2D Hann window (weighted K×K patch grid)
    hann1d = torch.hann_window(K, device=device, dtype=dtype)   # [K]
    hann2d = torch.outer(hann1d, hann1d)                               # [K, K]
    hann2d = hann2d / (hann2d.max().clamp_min(eps))    # normalized to [0,1]
    hann2d = hann2d[..., None]                                         # [K,K,1]

    # generate starting coordinates (ensure the right/bottom boundary is covered)
    def covered_starts(length, win, stride):
        starts = list(range(0, max(length - win, 0) + 1, stride))
        if not starts or starts[-1] + win < length:
            # append the last starting point to ensure the window covers the end
            starts.append(max(length - win, 0))
        return starts

    ys = covered_starts(H, window_size, stride)
    xs = covered_starts(W, window_size, stride)

    # pre-collect all tile coordinates (batch processing is more efficient)
    coords = []
    for y in ys:
        for x in xs:
            coords.append((y, x))

    # auxiliary: crop and pad the tile with window_size on the original image (reflect padding)
    def crop_reflect_pad(img_bchw, y, x, ws):
        y0, x0 = y, x
        y1, x1 = y + ws, x + ws

        y0_cl, x0_cl = max(0, y0), max(0, x0)
        y1_cl, x1_cl = min(H, y1), min(W, x1)

        tile = img_bchw[:, :, y0_cl:y1_cl, x0_cl:x1_cl]  # [B,3,hc,wc]

        pad_top    = y0_cl - y0        # >=0
        pad_left   = x0_cl - x0
        pad_bot    = y1 - y1_cl
        pad_right  = x1 - x1_cl

        if any(v > 0 for v in (pad_left, pad_right, pad_top, pad_bot)):
            tile = F.pad(tile, (pad_left, pad_right, pad_top, pad_bot), mode='reflect')

        return tile

    # main loop: batch process the tiles
    for start in range(0, len(coords), tile_batch):
        batch_coords = coords[start:start + tile_batch]
        tiles = []
        owners = []   
        for (y, x) in batch_coords:
            # crop/pad the tiles for all B images at this position
            # get tiles_b: [B,3,ws,ws]
            tiles_b = crop_reflect_pad(image, y, x, window_size)
            tiles.append(tiles_b)
            owners.append((y, x))

        # [num_tiles, B, 3, ws, ws] -> [num_tiles*B, 3, ws, ws]
        tiles = torch.cat(tiles, dim=0)  # [num_tiles*B, 3, ws, ws]

        # encode: output [num_tiles*B, K, K, D]
        patch_feats = encode_fn(tiles)
        assert patch_feats.dim() == 4 and patch_feats.shape[1] == K and patch_feats.shape[2] == K, \
            "encode_fn must return [N, K, K, D] format (without CLS)"
        N_tot, K1, K2, D = patch_feats.shape
        assert K1 == K2 == K

        if feat_canvas.shape[-1] == 0:
            # Update dtype to match patch_feats dtype
            dtype = patch_feats.dtype
            feat_canvas = torch.zeros(B, H_token, W_token, D, device=device, dtype=dtype)
            weight_canvas = weight_canvas.to(dtype=dtype)
            hann2d = hann2d.to(dtype=dtype)

        patch_feats = patch_feats * hann2d  # [N_tot, K, K, D]

        # restore to [num_tiles, B, K, K, D]
        num_tiles = len(batch_coords)
        patch_feats = patch_feats.view(num_tiles, B, K, K, D)

        print(f"num_tiles: {num_tiles}")
        # write each tile back to the corresponding global canonical grid position
        for t_idx, (y, x) in enumerate(batch_coords):
            u0 = (y // patch_size)
            v0 = (x // patch_size)
            u1 = min(u0 + K, H_token)
            v1 = min(v0 + K, W_token)
            uh = u1 - u0
            vh = v1 - v0

            # get the patch feats for all batch images of the current tile
            F_blk = patch_feats[t_idx, :, :uh, :vh, :]    # [B, uh, vh, D]
            W_blk = hann2d[:uh, :vh, :].expand(B, uh, vh, 1)    # [B, uh, vh, 1]

            # accumulate to the global canvas
            feat_canvas[:, u0:u1, v0:v1, :]  += F_blk
            weight_canvas[:, u0:u1, v0:v1, :] += W_blk

    # normalize
    feat_canvas = feat_canvas / (weight_canvas + eps)

    return feat_canvas  # [B, H_token, W_token, D]