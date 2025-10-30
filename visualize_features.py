from test_qwen25_vl import load_qwen_model
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pdb 

def visualize_feature_pca(features, feat_size, save_path='feature.png'):
    h, w = feat_size
    features_np = features.float().squeeze(0).detach().cpu().numpy()
    
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
    plt.title(f'Feature Map Visualization ({size}x{size})')
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
    hidden_states = {}

    def get_hook(name):
        def hook(module, input, output):
            hidden_states[name] = output.detach().cpu()  # 保存到CPU，释放显存
        return hook

    handles = []
    for i, blk in enumerate(model.model.layers):  # 对应 Qwen2.5-VL-3B
        if i in target_layers:
            handles.append(blk.register_forward_hook(get_hook(f"layer_{i}")))
    return hidden_states, handles


if __name__ == "__main__":
    image_path = "/data1/pinci/datasets/zoom_eye_data/vstar/relative_position/sa_6183.jpg"
    question = "Is the motorcycle on the left or right side of the dog?"

    model_path = "/data1/pinci/ckpt/huggingface//Qwen2.5-VL-3B-Instruct"
    model, processor, tokenizer = load_qwen_model(model_path, multi_gpu=True)

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

    # hidden_states 现在是字典：{ 'layer_4': tensor[B,S,D], ... }
    for name, feats in hidden_states.items():
        visual_feats = feats[:, v_token_indices, :].to("cpu")  # 取视觉 token
        print(name, visual_feats.shape)