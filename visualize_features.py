from test_qwen25_vl import load_qwen_model
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pdb 
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

def visualize_feature_pca(features, feat_size, save_path='feature.png'):
    h, w = feat_size
    # features_np = features.float().squeeze(0).detach().cpu().numpy()
    if features.dim() > 2:
        features_np = features.float().squeeze(0).numpy()
    else:
        features_np = features.float().numpy()
    
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(features_np)
    
    median = np.median(feat_pca, axis=0)
    q1 = np.percentile(feat_pca, 25, axis=0)
    q3 = np.percentile(feat_pca, 75, axis=0)
    iqr = q3 - q1
    scaled = (feat_pca - median) / (iqr + 1e-6)
    feat_pca_norm = 0.5 * (np.tanh(scaled) + 1)
    
    # size = int(np.sqrt(features_np.shape[0]))  # target_size//16
    # rgb_image = feat_pca_norm.reshape(size, size, 3)
    rgb_image = feat_pca_norm.reshape(h, w, 3)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(f'Feature Map Visualization ({h}x{w})')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return rgb_image

def get_visual_token_indices(input_ids, tokenizer):
    DEFAULT_IMAGE_TOKEN = 151655
    # DEFAULT_IMAGE_ST = 151652
    # DEFAULT_IMAGE_ED = 151653
    # cand_ids = [DEFAULT_IMAGE_ST, DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_ED]
    cand_ids = [DEFAULT_IMAGE_TOKEN]

    ids = input_ids[0].tolist()  # [S_text]
    hits = [i for i, tok in enumerate(ids) if tok in cand_ids]
    return hits if len(hits) > 0 else None

def register_hooks(model, target_layers):
    """
    在 Transformer 解码层上注册 forward hook，兼容单卡/多卡以及不同模型封装。
    返回收集到的 hidden_states 字典和 hook 句柄列表。
    """
    hidden_states = {}
    handles = []

    # 如果是 DataParallel 或 DDP，真实模型在 model.module 中
    base_model = model.module if hasattr(model, "module") else model

    # Qwen2_5_VLForConditionalGeneration 将 Qwen2_5_VLModel 存在 .model 属性
    qwen_model = getattr(base_model, "model", base_model)

    # 有的版本还会把解码层挂在 language_model.layers 下
    if hasattr(qwen_model, "layers"):
        layers = qwen_model.layers
    elif hasattr(qwen_model, "language_model") and hasattr(qwen_model.language_model, "layers"):
        layers = qwen_model.language_model.layers
    else:
        raise AttributeError(
            "无法在模型中找到解码层列表，请检查模型版本或修改上述逻辑。"
        )

    def get_hook(name):
        def hook(module, inputs, output):
            # 将输出 detatch 并转到 CPU，避免显存占用
            hidden_states[name] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
        return hook

    for i, blk in enumerate(layers):
        if i in target_layers:
            handles.append(blk.register_forward_hook(get_hook(f"layer_{i}")))

    return hidden_states, handles


def get_mllm_feat(model, processor, tokenizer, image_path, question):
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
    v_token_indices = get_visual_token_indices(question_inputs.input_ids, tokenizer)
    image_size = question_inputs.image_grid_thw[0]
    v_token_h, v_token_w = image_size[1] // 2, image_size[2] // 2

    total_layers = 37
    target_layers = list(range(4, total_layers, 4))
    hidden_states, handles = register_hooks(model, target_layers)

    with torch.no_grad():
        _ = model(
            input_ids=question_inputs.input_ids,
            attention_mask=question_inputs.attention_mask,
            pixel_values=question_inputs.pixel_values,
            image_grid_thw=question_inputs.image_grid_thw,
            return_dict=True,
        )

    for h in handles:  # 清除 hook，防止多次 forward 累积
        h.remove()

    for name, feats in hidden_states.items():
        visual_feats = feats[:, v_token_indices, :].to("cpu")
        visualize_feature_pca(visual_feats, (v_token_h, v_token_w), save_path=f"{name}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pca visualization")
    parser.add_argument("--annotation_path", type=str, default="/data1/pinci/datasets/zoom_eye_data/", help="Path to dataset root (contains benchmark folders)")
    parser.add_argument("--benchmark", type=str, choices=["vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld"], default="vstar")
    parser.add_argument('--dinov3_feat_path', type=str, default="/data1/pinci/datasets/zoom_eye_data/dinov3_features/vstar/dinov3_mllm_patch_28.pt")
    parser.add_argument('--vis_dir', type=str, default='./vis/pca_visualization/dinov3')
    args = parser.parse_args()

    dinov3_feat = torch.load(args.dinov3_feat_path, map_location='cpu')
    annotation_file = os.path.join(args.annotation_path, f"{args.benchmark}/annotation_{args.benchmark}.json")
    with open(annotation_file, 'r') as f:
        all_annotations = json.load(f)
    image_folder = os.path.join(args.annotation_path, f"{args.benchmark}")

    for annotation in tqdm(all_annotations):
        input_image = annotation['input_image']
        dinov3_patch_token = dinov3_feat[input_image]["patch_token"]
        resized_resolution = dinov3_feat[input_image]["resized_resolution"]
        v_token_h = resized_resolution[0] // 16
        v_token_w = resized_resolution[1] // 16
        save_path = os.path.join(args.vis_dir, input_image)
        visualize_feature_pca(dinov3_patch_token, (v_token_h, v_token_w), save_path=save_path)


    # image_path = "/data1/pinci/datasets/zoom_eye_data/vstar/relative_position/sa_6183.jpg"
    # question = "Is the motorcycle on the left or right side of the dog?"
    # model_path = "/data1/pinci/ckpt/huggingface/Qwen2.5-VL-3B-Instruct"
    # model, processor, tokenizer = load_qwen_model(model_path, multi_gpu=True)
    # get_mllm_feat(model, processor, tokenizer, image_path, question)